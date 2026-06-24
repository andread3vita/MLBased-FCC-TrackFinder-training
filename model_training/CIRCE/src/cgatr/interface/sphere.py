"""CGA sphere embedding in IPNS (Inner Product Null Space) representation.

In CGA Cl(4,1), a sphere with center c and radius ρ is a grade-1 vector:
    S = P(c) - (ρ²/2) · e∞

where P(c) is the null point for center c, and e∞ = e+ + e- is the point at infinity.

Properties:
    - S·S = -ρ² (non-null, unlike points)
    - For a point P: ⟨P, S⟩ = -(d² - ρ²)/2
      where d = distance from P to sphere center
    - ⟨P, S⟩ = 0 iff P lies on the sphere

Grade-1 indices: [1, 2, 3, 4, 5]
"""

import torch

from src.cgatr.interface.point import embed_point


def embed_sphere(center: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """Embed a sphere as a CGA IPNS grade-1 vector.

    S = P(center) - (radius²/2) · e∞

    Parameters
    ----------
    center : torch.Tensor with shape (..., 3)
        Sphere center coordinates (x, y, z).
    radius : torch.Tensor with shape (..., 1) or (...,)
        Sphere radius.

    Returns
    -------
    multivector : torch.Tensor with shape (..., 32)
        CGA sphere (grade-1 vector, non-null).
    """
    mv = embed_point(center)  # Start with null point P(center)

    r_sq = radius.squeeze(-1) ** 2 if radius.dim() > center.dim() - 1 else radius ** 2

    # Subtract (ρ²/2) · e∞ = (ρ²/2) · (e+ + e-)
    mv[..., 4] = mv[..., 4] - r_sq / 2  # e+ component
    mv[..., 5] = mv[..., 5] - r_sq / 2  # e- component

    return mv
