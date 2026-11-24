"""Opt-in ``torch.compile`` of the GATr network for GPU training and validation.

``torch.compile`` fuses the many small elementwise kernels of the equivariant
layers (linear maps, layer norms, gated nonlinearities, geometric products and
joins) between the attention calls. Attention itself stays eager: gatr marks
it ``@torch.compiler.disable`` because xFormers' ``memory_efficient_attention``
cannot be traced.

Static shapes with padding buckets: torch 2.2 mis-compiles GATr with symbolic
hit counts (``dynamic=True`` fails in Triton codegen on 1/sqrt(channels);
automatic dynamic shapes bake one batch's hit count into a stride of the
compiled backward, "assert_size_stride ... stride 14402==18113"). So the hit
count is padded up to one of ``buckets_per_octave`` sizes per factor of two
(``pad_inputs``) and every bucket compiles once with static shapes. The
padding rows copy the last real hit, form their own attention block and are
dropped from the output; all other GATr operations act per hit, so the real
hits' outputs and gradients are unchanged.

The xFormers ``BlockDiagonalMask`` of a batch is not passed through the
compiled graph. Dynamo would guard on its Python lists of sequence lengths and
recompile for every batch composition. Instead the mask is stored here
(``set_attention_mask``) and picked up by the eager attention call, while the
compiled network receives ``attention_mask=None``. The mask stays set until the
next batch replaces it, so gradient-checkpointing recomputation during the
backward pass sees the same mask as the forward pass.

Numerics: the same operations in FP32; only kernel fusion and accumulation
order change. Weights, ``state_dict`` keys and checkpoints are unchanged
(``nn.Module.compile`` compiles in place). Requires the CUDA toolkit's
``ptxas`` for Triton on GPU (``module load cuda/12.1`` on MN5).
"""

from __future__ import annotations

import math

import torch
import torch._dynamo
import torch.backends.opt_einsum
from torch import nn

import gatr.primitives.attention as gatr_attention
from gatr.utils.einsum import enable_cached_einsum

COMPILE_MODES = ("default", "max-autotune-no-cudagraphs")
DEFAULT_BUCKETS_PER_OCTAVE = 4
# Compiled variants kept per frame: buckets x {train, no_grad}; ~40 buckets
# cover 100 to 100k hits at 4 per octave.
_CACHE_SIZE_LIMIT = 256

_ATTENTION_MASK = None
_UNPATCHED_SDPA = None


def _sdpa_with_stored_mask(query, key, value, attn_mask=None):
    if attn_mask is None:
        attn_mask = _ATTENTION_MASK
    return _UNPATCHED_SDPA(query, key, value, attn_mask=attn_mask)


def set_attention_mask(mask) -> None:
    """Attention mask used by the eager attention of the compiled network."""
    global _ATTENTION_MASK
    _ATTENTION_MASK = mask


def bucket_size(n: int, buckets_per_octave: int = DEFAULT_BUCKETS_PER_OCTAVE) -> int:
    """Smallest bucket >= n: multiples of 2**(floor(log2 n) - log2 buckets)."""
    if n <= 64:
        return 64
    step = max(1, 2 ** (int(math.floor(math.log2(n))) - int(math.log2(buckets_per_octave))))
    return -(-n // step) * step


def pad_inputs(multivectors, scalars, join_reference, sequence_lengths,
               buckets_per_octave: int = DEFAULT_BUCKETS_PER_OCTAVE):
    """Pad the hits to ``bucket_size``; return padded tensors and lengths.

    Padding rows repeat the last hit (and its join reference) and get their
    own block in the attention mask, so they never mix with real hits.
    """
    n = int(multivectors.shape[0])
    padding = bucket_size(n, buckets_per_octave) - n
    if padding == 0:
        return multivectors, scalars, join_reference, list(sequence_lengths)

    def pad(tensor):
        if tensor is None:
            return None
        last = tensor[-1:].detach()
        return torch.cat((tensor, last.expand(padding, *tensor.shape[1:])), dim=0)

    return pad(multivectors), pad(scalars), pad(join_reference), [*sequence_lengths, padding]


def warm_up(network: nn.Module, multivectors, scalars=None, join_reference=None) -> None:
    """Fill gatr's lazily loaded basis caches for this device/dtype eagerly.

    gatr loads its product/linear tables (``torch.load``) on first use per
    device and dtype; traced by Dynamo, that load fails ("Cannot copy out of
    meta tensor"). One eager forward on a few hits fills the caches first.
    """
    key = (multivectors.device, multivectors.dtype)
    warmed = network.__dict__.setdefault("_gatr_compile_warmed", set())
    if key in warmed:
        return
    global _ATTENTION_MASK
    rows = min(int(multivectors.shape[0]), 8)
    stored, _ATTENTION_MASK = _ATTENTION_MASK, None  # full attention over these rows
    try:
        with torch.no_grad():
            network.forward(  # plain forward: bypasses the compiled __call__
                multivectors[:rows],
                scalars=None if scalars is None else scalars[:rows],
                attention_mask=None,
                join_reference=None if join_reference is None else join_reference[:rows],
            )
    finally:
        _ATTENTION_MASK = stored
    warmed.add(key)


def compile_gatr(network: nn.Module, mode: str = "default", backend: str = "inductor") -> nn.Module:
    """Compile ``network`` in place (static shapes, see ``pad_inputs``); return it.

    ``backend="aot_eager"`` traces identically without code generation (for
    CPU tests of the capture; the CPU inductor of torch 2.6 fails on GATr).
    """
    global _UNPATCHED_SDPA
    if mode not in COMPILE_MODES:
        raise ValueError(f"compile mode must be one of {COMPILE_MODES}, got {mode!r}")
    # gatr's cached einsum passes precomputed contraction paths, which
    # torch.compile (torch 2.2) rejects; plain torch.einsum is equivalent.
    enable_cached_einsum(False)
    # torch.einsum asks opt_einsum for a contraction order computed from the
    # concrete sizes, which pins the hit count in every traced frame and
    # recompiles for each batch size (falling back to eager after 8). Left to
    # right is already the cheap order for gatr's equations (weights x basis
    # first, then the hits).
    torch.backends.opt_einsum.enabled = False
    if _UNPATCHED_SDPA is None:
        _UNPATCHED_SDPA = gatr_attention.scaled_dot_product_attention
        gatr_attention.scaled_dot_product_attention = _sdpa_with_stored_mask
    # Static shapes: one compilation per padding bucket (see pad_inputs).
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, _CACHE_SIZE_LIMIT)
    if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
        torch._dynamo.config.accumulated_cache_size_limit = max(
            torch._dynamo.config.accumulated_cache_size_limit, 4 * _CACHE_SIZE_LIMIT
        )
    if backend == "inductor":
        network.compile(dynamic=False, mode=mode)
    else:
        network.compile(dynamic=False, backend=backend)
    return network

