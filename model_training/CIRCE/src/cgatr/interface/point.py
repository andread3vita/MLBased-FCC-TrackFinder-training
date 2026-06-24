"""CGA null vector point embedding.

In CGA Cl(4,1), a 3D point P(x,y,z) is embedded as a null grade-1 vector:
    P = x*e1 + y*e2 + z*e3 + ((1-r^2)/2)*e+ + ((1+r^2)/2)*e-
where r^2 = x^2 + y^2 + z^2.

The null condition P·P = 0 holds by construction.
The CGA distance is: d^2(P1, P2) = -2 <P1, P2>

Grade-1 indices in our ordering: [1, 2, 3, 4, 5]
  index 1: e1 (x)
  index 2: e2 (y)
  index 3: e3 (z)
  index 4: e+ = (e4)
  index 5: e- = (e5)
"""

import torch


def embed_point(coordinates: torch.Tensor) -> torch.Tensor:
    """Embeds 3D points as CGA null vectors (grade-1).

    Parameters
    ----------
    coordinates : torch.Tensor with shape (..., 3)
        3D coordinates (x, y, z).

    Returns
    -------
    multivector : torch.Tensor with shape (..., 32)
        CGA null vector embedding.
    """
    batch_shape = coordinates.shape[:-1]
    mv = torch.zeros(*batch_shape, 32, dtype=coordinates.dtype, device=coordinates.device)

    x = coordinates[..., 0]
    y = coordinates[..., 1]
    z = coordinates[..., 2]
    r_sq = x * x + y * y + z * z

    # Grade-1 components
    mv[..., 1] = x       # e1
    mv[..., 2] = y       # e2
    mv[..., 3] = z       # e3
    mv[..., 4] = (1.0 - r_sq) / 2.0  # e+
    mv[..., 5] = (1.0 + r_sq) / 2.0  # e-

    return mv


def extract_point(multivector: torch.Tensor, threshold: float = 1e-3) -> torch.Tensor:
    """Extract 3D coordinates from a CGA null vector.

    For a CGA point P = x*e1 + y*e2 + z*e3 + a*e+ + b*e-,
    the Euclidean coordinates are (e1, e2, e3) / e- = (x/b, y/b, z/b).

    Parameters
    ----------
    multivector : torch.Tensor with shape (..., 32)
    threshold : float
        Minimum value of e- component to avoid division by zero.

    Returns
    -------
    coordinates : torch.Tensor with shape (..., 3)
    """
    e_minus = multivector[..., [5]]  # e- component
    e_minus = torch.where(torch.abs(e_minus) > threshold, e_minus, torch.full_like(e_minus, threshold))

    coordinates = torch.cat([
        multivector[..., [1]],  # e1 -> x
        multivector[..., [2]],  # e2 -> y
        multivector[..., [3]],  # e3 -> z
    ], dim=-1) / e_minus

    return coordinates
