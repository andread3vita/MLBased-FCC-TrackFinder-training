"""Inner product and invariants for CGA Cl(4,1).

In CGA Cl(4,1), the metric is non-degenerate: ALL 32 components contribute to the
inner product. This is a key difference from PGA Cl(3,0,1) where degenerate e0
components were excluded.

The inner product is: <x, y> = <~x * y>_0 (grade-0 part of reverse(x) * y).
For non-degenerate algebras, this simplifies to a weighted dot product with signs
from the reversal and metric.
"""

import torch

from src.cgatr.primitives.linear import _compute_reversal, grade_project


def compute_inner_product_mask(gp, device=torch.device("cpu")) -> torch.Tensor:
    """Compute inner product mask from GP table.

    For non-degenerate Cl(4,1), ALL components contribute (unlike PGA).
    The diagonal of gp[0] (scalar component of blade_i * blade_i) gives the metric,
    combined with reversal signs.

    Parameters
    ----------
    gp : torch.Tensor with shape (32, 32, 32)
    device : torch.device

    Returns
    -------
    ip_weights : torch.Tensor with shape (32,)
        Weights for inner product: ip(x,y) = sum_i weights_i * x_i * y_i
    """
    reversal = _compute_reversal(device=device, dtype=torch.float32)
    # gp[0, i, i] gives the scalar component of blade_i * blade_i
    metric_diag = torch.diag(gp[0].to(device))
    ip_weights = reversal * metric_diag
    return ip_weights


def compute_inner_product_indices(gp, device=torch.device("cpu")) -> torch.Tensor:
    """Compute which indices contribute to inner product (nonzero metric).

    For Cl(4,1) all 32 components contribute.

    Returns
    -------
    indices : torch.Tensor
        Indices where inner product weight is nonzero.
    """
    weights = compute_inner_product_mask(gp, device=device)
    return torch.arange(32, device=device)[weights.abs() > 1e-10]


def inner_product(
    ip_weights: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    channel_sum: bool = False,
) -> torch.Tensor:
    """Computes the CGA inner product: <x, y> = <~x * y>_0.

    For non-degenerate Cl(4,1), this is a weighted dot product using all 32 components.

    Parameters
    ----------
    ip_weights : torch.Tensor with shape (32,)
        Inner product weights (reversal * metric diagonal).
    x : torch.Tensor with shape (..., 32) or (..., channels, 32)
    y : torch.Tensor with shape (..., 32) or (..., channels, 32)
    channel_sum : bool
        Whether to sum over the channel dimension.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 1)
    """
    # Weight both inputs
    weighted = ip_weights.to(x.device, x.dtype) * x * y

    if channel_sum:
        outputs = torch.sum(torch.sum(weighted, dim=-1), dim=-1)
    else:
        outputs = torch.sum(weighted, dim=-1)

    return outputs.unsqueeze(-1)


def norm(ip_weights: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """CGA norm: sqrt(|<x, x>|)."""
    return torch.sqrt(torch.clamp(torch.abs(inner_product(ip_weights, x, x)), 0.0))


def pin_invariants(ip_weights: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Computes 6 invariants: scalar component + norms of grades 1-5.

    Parameters
    ----------
    ip_weights : torch.Tensor with shape (32,)
    x : torch.Tensor with shape (..., 32)

    Returns
    -------
    outputs : torch.Tensor with shape (..., 6)
    """
    projections = grade_project(x)  # (..., 6, 32)
    squared_norms = inner_product(ip_weights, projections, projections)[..., 0]  # (..., 6)
    norms = torch.sqrt(torch.clamp(torch.abs(squared_norms), 0.0))
    return torch.cat((x[..., [0]], norms[..., 1:]), dim=-1)  # (..., 6)
