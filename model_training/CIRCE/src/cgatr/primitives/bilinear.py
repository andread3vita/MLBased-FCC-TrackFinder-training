"""Bilinear operations for CGA Cl(4,1): geometric product and outer product.

All operations use 32-component multivectors and (32, 32, 32) Cayley tables.
"""

import torch
from torch import nn


class geometric_product(nn.Module):
    """Geometric product using precomputed Cayley table."""

    def __init__(self, gp) -> None:
        super().__init__()
        self.register_buffer("gp", gp)  # (32, 32, 32)

    def forward(self, x, y):
        # x, y: (..., 32). Arbitrary leading dims supported via ellipsis
        # so the same op handles single-event (items, channels, 32) and
        # multi-event batched (B, N, channels, 32) inputs without a
        # custom rank-2 prefix.
        # Two-step einsum for ONNX compatibility.
        outputs1 = torch.einsum("ijk, ...j -> ...ik", self.gp, x)
        outputs = torch.einsum("...ik, ...k -> ...i", outputs1, y)
        return outputs


def outer_product(op, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the outer (wedge) product f(x,y) = x ^ y.

    Parameters
    ----------
    op : torch.Tensor with shape (32, 32, 32)
        Outer product Cayley table.
    x : torch.Tensor with shape (..., 32)
        First input multivector. Arbitrary leading dims supported.
    y : torch.Tensor with shape (..., 32)
        Second input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 32)
        Wedge product result.
    """
    outputs1 = torch.einsum("ijk, ...j -> ...ik", op, x)
    outputs = torch.einsum("...ik, ...k -> ...i", outputs1, y)
    return outputs
