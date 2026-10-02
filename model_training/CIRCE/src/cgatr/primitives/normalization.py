"""Equivariant normalization for CGA Cl(4,1)."""

import torch

from src.cgatr.primitives.invariants import inner_product

# Contiguous blade-index ranges per grade, keyed by blade count. The grade
# partition is a property of the algebra, so the blade count identifies it
# unambiguously for the two algebras in play and saves threading it through
# every layer.
#   32 -> CGA Cl(4,1),    grades 0..5, dims 1,5,10,10,5,1
#   16 -> PGA Cl(3,0,1),  grades 0..4, dims 1,4,6,4,1
_GRADE_RANGES_BY_BLADES = {
    32: ((0, 1), (1, 6), (6, 16), (16, 26), (26, 31), (31, 32)),
    16: ((0, 1), (1, 5), (5, 11), (11, 15), (15, 16)),
}


def equi_layer_norm(
    ip_weights: torch.Tensor,
    x: torch.Tensor,
    channel_dim: int = -2,
    gain: float = 1.0,
    epsilon: float = 0.01,
    gradewise: bool = False,
    epsilon_mode: str = "clamp",
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
    epsilon_mode : {"clamp", "add"}
        ``clamp`` reproduces the released GATr code, ``max(norm², epsilon)``.
        ``add`` follows the literal C-GATr equation, ``norm² + epsilon``.

    Returns
    -------
    outputs : torch.Tensor with shape (batch, channels, 32)
    """
    if gradewise:
        wsq = ip_weights.to(x.dtype) * x * x  # (..., channels, 32)
        squared_norms = None
        for a, b in _GRADE_RANGES_BY_BLADES[x.shape[-1]]:
            g = wsq[..., a:b].sum(dim=-1, keepdim=True).abs()  # (..., channels, 1)
            squared_norms = g if squared_norms is None else squared_norms + g
    else:
        squared_norms = inner_product(ip_weights, x, x)  # (..., channels, 1)
        # Take absolute value since CGA inner product can be negative
        squared_norms = torch.abs(squared_norms)
    squared_norms = torch.mean(squared_norms, dim=channel_dim, keepdim=True)
    if epsilon_mode == "clamp":
        squared_norms = torch.clamp(squared_norms, min=epsilon)
    elif epsilon_mode == "add":
        squared_norms = squared_norms + epsilon
    else:
        raise ValueError(f"unknown LayerNorm epsilon mode: {epsilon_mode!r}")
    outputs = gain * x / torch.sqrt(squared_norms)
    return outputs
