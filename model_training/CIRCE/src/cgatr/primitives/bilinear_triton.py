"""Sparse FP32 Triton kernel for a 32-blade Clifford geometric product.

Each pair of basis blades contributes to exactly one output blade.  The dense
Cayley tensor therefore has 32**2 nonzero entries out of 32**3.  This kernel
evaluates only those structural nonzeros and uses one warp per multivector
pair.  The generated tables use a grade-grouped blade ordering, so output
indices are precomputed rather than assuming bit-mask / XOR ordering.

This module is deliberately optional.  The production reference remains the
dense PyTorch implementation until forward, backward, model-level numerical,
equivariance, memory, and throughput gates all pass on the target GPU.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only without Triton
    triton = None
    tl = None


NUM_BLADES = 32


if triton is not None:

    @triton.jit
    def _sparse_gp_forward_kernel(
        x_ptr,
        y_ptr,
        forward_left_ptr,
        forward_coeff_ptr,
        out_ptr,
        x_item_stride,
        y_item_stride,
        channels: tl.constexpr,
    ):
        row = tl.program_id(0)
        item = row // channels
        channel = row - item * channels
        blade = tl.arange(0, 32)
        x_base = item * x_item_stride + channel * 32
        y_base = item * y_item_stride + channel * 32
        out_base = row * 32
        acc = tl.zeros((32,), dtype=tl.float32)

        # Match the reference's final contraction axis: k increases from 0..31.
        # For every (output blade i, right blade k), the table gives the unique
        # contributing left blade j and Cayley coefficient.
        for k in tl.static_range(0, 32):
            table_offset = blade * 32 + k
            j = tl.load(forward_left_ptr + table_offset)
            x = tl.load(x_ptr + x_base + j)
            y = tl.load(y_ptr + y_base + k)
            coeff = tl.load(forward_coeff_ptr + table_offset)
            acc += coeff * x * y

        tl.store(out_ptr + out_base + blade, acc)


    @triton.jit
    def _sparse_gp_backward_kernel(
        grad_out_ptr,
        x_ptr,
        y_ptr,
        pair_output_ptr,
        pair_coeff_ptr,
        grad_x_ptr,
        grad_y_ptr,
        x_item_stride,
        y_item_stride,
        channels: tl.constexpr,
    ):
        row = tl.program_id(0)
        item = row // channels
        channel = row - item * channels
        blade = tl.arange(0, 32)
        x_base = item * x_item_stride + channel * 32
        y_base = item * y_item_stride + channel * 32
        grad_base = row * 32
        grad_x = tl.zeros((32,), dtype=tl.float32)
        grad_y = tl.zeros((32,), dtype=tl.float32)

        # lane `blade` is j for grad_x and k for grad_y.  Compute both gradients
        # in one launch so x, y, grad_out, and the Cayley tables are reused.
        for other in tl.static_range(0, 32):
            pair_x = blade * 32 + other
            out_x = tl.load(pair_output_ptr + pair_x)
            coeff_x = tl.load(pair_coeff_ptr + pair_x)
            grad_out_x = tl.load(grad_out_ptr + grad_base + out_x)
            y = tl.load(y_ptr + y_base + other)
            grad_x += coeff_x * grad_out_x * y

            pair_y = other * 32 + blade
            out_y = tl.load(pair_output_ptr + pair_y)
            coeff_y = tl.load(pair_coeff_ptr + pair_y)
            grad_out_y = tl.load(grad_out_ptr + grad_base + out_y)
            x = tl.load(x_ptr + x_base + other)
            grad_y += coeff_y * grad_out_y * x

        tl.store(grad_x_ptr + grad_base + blade, grad_x)
        tl.store(grad_y_ptr + grad_base + blade, grad_y)


def cayley_sparse_tables(
    gp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract index and coefficient tables for the structurally sparse product."""
    if gp.shape != (NUM_BLADES, NUM_BLADES, NUM_BLADES):
        raise ValueError(f"expected a 32x32x32 Cayley tensor, got {tuple(gp.shape)}")

    # pair_output[j,k] and pair_coeff[j,k] describe e_j e_k.
    nonzero_per_pair = torch.count_nonzero(gp, dim=0)
    if not torch.equal(nonzero_per_pair, torch.ones_like(nonzero_per_pair)):
        raise ValueError("Cayley tensor does not have one output per blade pair")
    pair_output = gp.abs().argmax(dim=0).to(torch.int32).contiguous()
    pair_coeff = torch.gather(
        gp, 0, pair_output.to(torch.long).unsqueeze(0)
    ).squeeze(0).contiguous()

    # For fixed right blade k, left multiplication is a permutation because
    # Cl(4,1) is non-degenerate.  Invert it for coalesced forward output.
    forward_left = torch.empty_like(pair_output)
    forward_coeff = torch.empty_like(pair_coeff)
    left = torch.arange(NUM_BLADES, device=gp.device, dtype=torch.int32)
    for k in range(NUM_BLADES):
        outputs = pair_output[:, k]
        if torch.unique(outputs).numel() != NUM_BLADES:
            raise ValueError("right multiplication is not a blade permutation")
        forward_left[outputs, k] = left
        forward_coeff[outputs, k] = pair_coeff[:, k]

    return (
        pair_output,
        pair_coeff,
        forward_left.contiguous(),
        forward_coeff.contiguous(),
    )


class _SparseGeometricProductFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pair_output: torch.Tensor,
        pair_coeff: torch.Tensor,
        forward_left: torch.Tensor,
        forward_coeff: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        if triton is None:
            raise RuntimeError("Triton is required for the sparse CGA kernel")
        tables = (pair_output, pair_coeff, forward_left, forward_coeff)
        if not (x.is_cuda and y.is_cuda and all(table.is_cuda for table in tables)):
            raise RuntimeError("sparse CGA kernel requires CUDA tensors")
        if x.dtype != torch.float32 or y.dtype != torch.float32:
            raise RuntimeError("sparse CGA kernel is intentionally FP32-only")
        if x.shape != y.shape or x.shape[-1] != NUM_BLADES:
            raise ValueError(
                f"expected matching (...,32) inputs, got {x.shape} and {y.shape}"
            )
        if x.stride(-1) != 1 or y.stride(-1) != 1:
            raise RuntimeError("sparse CGA kernel requires contiguous blade axes")
        if x.stride(-2) != NUM_BLADES or y.stride(-2) != NUM_BLADES:
            raise RuntimeError("sparse CGA kernel requires packed channel axes")
        if not all(table.is_contiguous() for table in tables):
            raise RuntimeError("sparse CGA lookup tables must be contiguous")

        out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        channels = x.shape[-2]
        items = x.numel() // (channels * NUM_BLADES)
        rows = items * channels
        x_item_stride = x.stride(-3) if x.ndim >= 3 else channels * NUM_BLADES
        y_item_stride = y.stride(-3) if y.ndim >= 3 else channels * NUM_BLADES
        _sparse_gp_forward_kernel[(rows,)](
            x,
            y,
            forward_left,
            forward_coeff,
            out,
            x_item_stride,
            y_item_stride,
            channels=channels,
            num_warps=1,
        )
        ctx.save_for_backward(pair_output, pair_coeff, x, y)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        pair_output, pair_coeff, x, y = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_x = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        grad_y = torch.empty(y.shape, device=y.device, dtype=y.dtype)
        channels = x.shape[-2]
        items = x.numel() // (channels * NUM_BLADES)
        rows = items * channels
        x_item_stride = x.stride(-3) if x.ndim >= 3 else channels * NUM_BLADES
        y_item_stride = y.stride(-3) if y.ndim >= 3 else channels * NUM_BLADES
        _sparse_gp_backward_kernel[(rows,)](
            grad_out,
            x,
            y,
            pair_output,
            pair_coeff,
            grad_x,
            grad_y,
            x_item_stride,
            y_item_stride,
            channels=channels,
            num_warps=1,
        )
        return None, None, None, None, grad_x, grad_y


def sparse_geometric_product(
    pair_output: torch.Tensor,
    pair_coeff: torch.Tensor,
    forward_left: torch.Tensor,
    forward_coeff: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Apply the optional sparse FP32 geometric-product implementation."""
    return _SparseGeometricProductFn.apply(
        pair_output,
        pair_coeff,
        forward_left,
        forward_coeff,
        x,
        y,
    )
