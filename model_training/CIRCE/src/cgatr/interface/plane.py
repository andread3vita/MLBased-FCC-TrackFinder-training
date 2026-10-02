"""CGA plane embedding in IPNS (Inner Product Null Space) representation.

A plane with unit normal n at signed distance d from the origin is

    pi = n + d inf

with inf = e- - e+ the point at infinity, which in our blade ordering is
index 5 minus index 4. See `src/cgatr/interface/sphere.py` for the convention and
for why this is easy to get wrong: the original code here read "inf = e+ + e-"
and wrote +d into both components, offsetting along 2o, the origin, instead.

Since <P, 2o> = -|p|^2 rather than the constant <P, inf> = -1, that gives

    <P, pi> = n.p - d |p|^2      instead of  n.p - d

so the locus where the plane test vanishes is not a plane, and the embedding is
not translation-covariant (85% relative error for a 0.44 shift in embedded
units). `pi.pi = 1` holds either way, which is why a norm check does not catch
it.

`fix_null=False` reproduces the original behaviour and is the default, because
every conformal checkpoint trained so far used it through `embed_circle_ipns`.

Properties with `fix_null=True`:
    - pi.pi = 1 for a unit normal
    - <P, pi> = n.x - d for a point P at x, zero iff x lies on the plane

Grade-1 indices: [1, 2, 3, 4, 5]
"""

import torch


def embed_plane(
    normal: torch.Tensor,
    point_on_plane: torch.Tensor,
    fix_null: bool = False,
) -> torch.Tensor:
    """Embed a plane as a CGA IPNS grade-1 vector.

    Parameters
    ----------
    normal : torch.Tensor with shape (..., 3)
        Unit normal vector of the plane.
    point_on_plane : torch.Tensor with shape (..., 3)
        Any point on the plane.
    fix_null : bool
        Offset along inf = e- - e+ rather than 2o = e+ + e-. See the module
        docstring; False reproduces the original behaviour.

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

    # d = n . p (signed distance from origin)
    d = (normal * point_on_plane).sum(dim=-1)

    if fix_null:
        # pi = n + d inf,  inf = e- - e+
        mv[..., 4] = -d
        mv[..., 5] = d
    else:
        # pi = n + d (e+ + e-) = n + 2 d o, the original mistake
        mv[..., 4] = d
        mv[..., 5] = d

    return mv
