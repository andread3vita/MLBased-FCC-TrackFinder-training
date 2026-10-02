"""CGA circle embedding in IPNS representation.

A circle in CGA is the intersection of a sphere and a plane:
    C = S ∧ π     (grade-2 bivector)

where:
    S = drift sphere (center=wire position, radius=drift distance)
    π = constraint plane (perpendicular to wire, through wire position)

This is a single outer product of two grade-1 vectors, giving a grade-2 bivector.
Much cheaper than the OPNS representation (P1∧P2∧P3 = two outer products).

Properties:
    - C is a grade-2 bivector (indices [6..15], 10 components)
    - Points on the circle satisfy both ⟨P,S⟩=0 and ⟨P,π⟩=0

Grade-2 indices: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
"""

import torch

from src.cgatr.interface.sphere import embed_sphere
from src.cgatr.interface.plane import embed_plane
from src.cgatr.primitives.bilinear import outer_product


def embed_circle_ipns(
    wire_pos: torch.Tensor,
    wire_dir: torch.Tensor,
    drift_radius: torch.Tensor,
    op_table: torch.Tensor,
    fix_null: bool = False,
) -> torch.Tensor:
    """Embed a drift chamber circle as IPNS grade-2 bivector: C = S ∧ π.

    Parameters
    ----------
    wire_pos : torch.Tensor (..., 3)
        Wire center position.
    wire_dir : torch.Tensor (..., 3)
        Wire direction unit vector (normal to the constraint plane).
    drift_radius : torch.Tensor (...,) or (..., 1)
        Drift distance / radius.
    op_table : torch.Tensor (32, 32, 32)
        Outer product Cayley table.
    fix_null : bool
        Offset the sphere and plane along the point at infinity, as their
        definitions require, rather than along the origin. Only then does the
        circle mean what it claims and transform covariantly under translation.
        See `src/cgatr/interface/sphere.py`. False reproduces the behaviour every
        conformal checkpoint was trained with.

    Returns
    -------
    circle : torch.Tensor (..., 32)
        IPNS circle (grade-2 bivector).
    """
    S = embed_sphere(wire_pos, drift_radius, fix_null=fix_null)  # grade-1
    pi = embed_plane(wire_dir, wire_pos, fix_null=fix_null)  # grade-1

    # Circle = S ∧ π (single outer product → grade-2 bivector)
    S_ex = S.unsqueeze(-2)   # (..., 1, 32)
    pi_ex = pi.unsqueeze(-2)  # (..., 1, 32)
    circle = outer_product(op_table, S_ex, pi_ex)  # (..., 1, 32)

    return circle.squeeze(-2)  # (..., 32)


def embed_dc_two_channel(
    wire_pos: torch.Tensor,
    wire_dir: torch.Tensor,
    drift_radius: torch.Tensor,
    op_table: torch.Tensor,
    fix_null: bool = False,
) -> torch.Tensor:
    """Two-channel DC embedding: sphere (grade-1) + circle (grade-2).

    Channel 0: drift sphere — same grade as VTX points, enables direct attention
    Channel 1: IPNS circle — carries full geometric constraint

    Parameters
    ----------
    wire_pos, wire_dir, drift_radius, op_table : see embed_circle_ipns

    Returns
    -------
    mv : torch.Tensor (..., 2, 32)
        Two-channel multivector.
    """
    sphere = embed_sphere(wire_pos, drift_radius, fix_null=fix_null)  # grade-1
    circle = embed_circle_ipns(wire_pos, wire_dir, drift_radius, op_table,
                               fix_null=fix_null)  # grade-2

    return torch.stack([sphere, circle], dim=-2)  # (..., 2, 32)


def embed_vtx_two_channel(pos: torch.Tensor) -> torch.Tensor:
    """Two-channel VTX embedding: point (grade-1) + zero padding.

    Channel 0: CGA null point — grade-1
    Channel 1: zeros — padding to match DC's 2 channels

    Parameters
    ----------
    pos : torch.Tensor (..., 3)

    Returns
    -------
    mv : torch.Tensor (..., 2, 32)
    """
    from src.cgatr.interface.point import embed_point

    point = embed_point(pos)  # (..., 32) grade-1
    zeros = torch.zeros_like(point)  # (..., 32)

    return torch.stack([point, zeros], dim=-2)  # (..., 2, 32)


def embed_circle_from_features(
    wire_x: torch.Tensor,
    wire_y: torch.Tensor,
    wire_z: torch.Tensor,
    drift_distance: torch.Tensor,
    azimuthal: torch.Tensor,
    stereo: torch.Tensor,
    op_table: torch.Tensor,
    two_channel: bool = True,
    fix_null: bool = False,
    fix_wire_dir: bool = False,
) -> torch.Tensor:
    """Embed DC hits from raw features.

    Parameters
    ----------
    wire_x, wire_y, wire_z : (N,) wire center coords
    drift_distance : (N,) drift radius
    azimuthal, stereo : (N,) wire angles
    op_table : (32, 32, 32) outer product table
    two_channel : bool
        If True, return (N, 2, 32) with sphere + circle channels.
        If False, return (N, 32) with just the IPNS circle.

    Returns
    -------
    mv : torch.Tensor (N, 2, 32) or (N, 32)
    """
    wire_pos = torch.stack([wire_x, wire_y, wire_z], dim=-1)

    # Wire direction from angles
    cos_s = torch.cos(stereo)
    sin_s = torch.sin(stereo)
    cos_a = torch.cos(azimuthal)
    sin_a = torch.sin(azimuthal)

    # A stereo wire tilts azimuthally, not radially; see CGATrParquetModel for
    # what the wrong slot ordering costs.
    if fix_wire_dir:
        wire_dir = torch.stack([
            sin_s * sin_a,
            -sin_s * cos_a,
            cos_s,
        ], dim=-1)
    else:
        wire_dir = torch.stack([
            sin_s * cos_a,
            sin_s * sin_a,
            cos_s,
        ], dim=-1)
    wire_dir = wire_dir / (torch.norm(wire_dir, dim=-1, keepdim=True) + 1e-8)

    if two_channel:
        return embed_dc_two_channel(wire_pos, wire_dir, drift_distance, op_table,
                                    fix_null=fix_null)
    else:
        return embed_circle_ipns(wire_pos, wire_dir, drift_distance, op_table,
                                 fix_null=fix_null)
