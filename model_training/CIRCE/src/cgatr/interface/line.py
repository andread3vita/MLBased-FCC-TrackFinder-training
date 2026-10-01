"""Conformal embedding of an oriented Euclidean line."""

import torch

from src.cgatr.interface.point import embed_point
from src.cgatr.primitives.bilinear import outer_product


def embed_line(
    position: torch.Tensor,
    direction: torch.Tensor,
    outer_product_table: torch.Tensor,
) -> torch.Tensor:
    """Embed a line as the OPNS trivector ``P(position) ^ d ^ infinity``.

    ``position`` may be any point on the line. ``direction`` is normalized here
    so that the otherwise homogeneous representation has a deterministic scale.
    Reversing the direction reverses the oriented line; the detector convention
    supplies one consistent orientation for every wire.
    """
    if position.shape[-1] != 3 or direction.shape[-1] != 3:
        raise ValueError("position and direction must end in three coordinates")
    direction = direction / direction.norm(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-8)

    point = embed_point(position)
    direction_mv = torch.zeros_like(point)
    direction_mv[..., 1:4] = direction
    infinity = torch.zeros_like(point)
    infinity[..., 4] = -1.0
    infinity[..., 5] = 1.0

    point_direction = outer_product(
        outer_product_table, point, direction_mv
    )
    return outer_product(outer_product_table, point_direction, infinity)
