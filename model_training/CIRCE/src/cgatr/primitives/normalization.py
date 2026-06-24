"""Equivariant normalization for CGA Cl(4,1)."""

import torch

from src.cgatr.primitives.invariants import inner_product

# Contiguous blade-index ranges per grade for CGA Cl(4,1) (grades 0..5;
# dims 1,5,10,10,5,1). From cga_metadata.pt grade_ranges.
_GRADE_RANGES = ((0, 1), (1, 6), (6, 16), (16, 26), (26, 31), (31, 32))


def equi_layer_norm(
    ip_weights: torch.Tensor,
    x: torch.Tensor,
    channel_dim: int = -2,
    gain: float = 1.0,
    epsilon: float = 0.01,
    gradewise: bool = False,
) -> torch.Tensor:
    """Equivariant LayerNorm using CGA inner product.

    Rescales input such that mean_channels |x|^2 = 1.

    Parameters
    ----------
    ip_weights : torch.Tensor with shape (32,)
        Inner product weights.
    x : torch.Tensor with shape (batch, channels, 32)
    channel_dim : int
    gain : float
    epsilon : float
    gradewise : bool
        If True, take abs of the inner product PER GRADE before summing across
        grades (de Haan et al. 2311.04744, Sec. 3.4 C-GATr recipe). Avoids null
        multivectors from +/- grade cancellation that otherwise let coefficients
        grow by 1/sqrt(epsilon) each layer. If False (default), uses the
        whole-multivector abs(<x,x>) (the paper intermediate, less stable).

    Returns
    -------
    outputs : torch.Tensor with shape (batch, channels, 32)
    """
    if gradewise:
        wsq = ip_weights.to(x.dtype) * x * x  # (..., channels, 32)
        squared_norms = None
        for a, b in _GRADE_RANGES:
            g = wsq[..., a:b].sum(dim=-1, keepdim=True).abs()  # (..., channels, 1)
            squared_norms = g if squared_norms is None else squared_norms + g
    else:
        squared_norms = inner_product(ip_weights, x, x)  # (..., channels, 1)
        # Take absolute value since CGA inner product can be negative
        squared_norms = torch.abs(squared_norms)
    squared_norms = torch.mean(squared_norms, dim=channel_dim, keepdim=True)
    squared_norms = torch.clamp(squared_norms, epsilon)
    outputs = gain * x / torch.sqrt(squared_norms)
    return outputs
