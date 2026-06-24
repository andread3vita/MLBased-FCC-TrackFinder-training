"""CGA plane embedding in IPNS (Inner Product Null Space) representation.

In CGA Cl(4,1), a plane with unit normal n at signed distance d from origin is:
    π = n + d · e∞

where e∞ = e+ + e- is the point at infinity.

Properties:
    - π·π = 1 (for unit normal n)
    - For a point P at position x: ⟨P, π⟩ = -(n·x - d)
      This is zero iff x lies on the plane.

A plane through point p with normal n has d = n·p, so:
    π = n + (n·p) · e∞

Grade-1 indices: [1, 2, 3, 4, 5]
"""

import torch


def embed_plane(normal: torch.Tensor, point_on_plane: torch.Tensor) -> torch.Tensor:
    """Embed a plane as a CGA IPNS grade-1 vector.

    π = n + (n·p) · e∞

    Parameters
    ----------
    normal : torch.Tensor with shape (..., 3)
        Unit normal vector of the plane.
    point_on_plane : torch.Tensor with shape (..., 3)
        Any point on the plane.

    Returns
    -------
    multivector : torch.Tensor with shape (..., 32)
        CGA plane (grade-1 vector).
    """
    batch_shape = normal.shape[:-1]
    mv = torch.zeros(*batch_shape, 32, dtype=normal.dtype, device=normal.device)

    # Normal components in e1, e2, e3
    mv[..., 1] = normal[..., 0]
    mv[..., 2] = normal[..., 1]
    mv[..., 3] = normal[..., 2]

    # d = n · p (signed distance from origin)
    d = (normal * point_on_plane).sum(dim=-1)

    # d · e∞ = d · (e+ + e-)
    mv[..., 4] = d  # e+ component
    mv[..., 5] = d  # e- component

    return mv
