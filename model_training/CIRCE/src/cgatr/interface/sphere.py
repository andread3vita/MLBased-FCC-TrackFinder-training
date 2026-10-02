"""CGA sphere embedding in IPNS (Inner Product Null Space) representation.

A sphere with centre c and radius r is the null point of the centre offset along
the point at infinity:

    S = P(c) - (r^2 / 2) inf

The subtlety is which blades `inf` occupies. Following EPC (arXiv:2311.04744)
Sec. 2, the two null vectors are

    inf = e- - e+          o = (e+ + e-) / 2

and `embed_point` writes exactly that convention, P = o + p + (|p|^2/2) inf. In
our blade ordering e+ is index 4 and e- is index 5, so

    inf = e5 - e4          2o = e4 + e5

The original code here read "inf = e+ + e-" and subtracted r^2/2 from *both*
components, which is -(r^2/2) * 2o: it offset the centre along the **origin**
rather than along infinity. That is the same o-versus-inf confusion that the
equivariant linear basis had before `--no_legacy_equivariance`, in a second and
independent place.

It matters because o and inf contract with a point quite differently:

    <P, inf> = -1          (constant, which is what an offset needs)
    <P, 2o>  = -|p|^2      (depends on where the point is)

So with the original direction the sphere is not the sphere it claims to be:

    <P,S> = -|p-c|^2/2 + r^2 |p|^2 / 2      instead of  -(|p-c|^2 - r^2)/2
    S.S   = +r^2 |c|^2                      instead of  +r^2

meaning <P,S> = 0 is not the point-on-sphere condition, the drift radius enters
weighted by the *probe* hit's squared distance from the origin, and the
embedding is not translation-covariant -- about 1% relative error for a 0.5
shift in embedded units for the sphere alone, and 40% once it is wedged into a
circle, since the plane carried the same error.

`fix_null=False` reproduces the original behaviour and is the default, because
every conformal checkpoint trained so far saw those inputs and has to be
evaluated on them. New runs should pass `fix_null=True`.

Properties with `fix_null=True`:
    - S.S = +r^2 (non-null, unlike points)
    - <P, S> = -(d^2 - r^2)/2 for d the distance from P to the centre
    - <P, S> = 0 iff P lies on the sphere

Grade-1 indices: [1, 2, 3, 4, 5]
"""

import torch

from src.cgatr.interface.point import embed_point


def embed_sphere(
    center: torch.Tensor,
    radius: torch.Tensor,
    fix_null: bool = False,
) -> torch.Tensor:
    """Embed a sphere as a CGA IPNS grade-1 vector.

    Parameters
    ----------
    center : torch.Tensor with shape (..., 3)
        Sphere center coordinates (x, y, z).
    radius : torch.Tensor with shape (..., 1) or (...,)
        Sphere radius.
    fix_null : bool
        Offset along the point at infinity, inf = e- - e+, as the definition
        requires, rather than along 2o = e+ + e-. See the module docstring;
        False reproduces the original behaviour.

    Returns
    -------
    multivector : torch.Tensor with shape (..., 32)
        CGA sphere (grade-1 vector, non-null).
    """
    mv = embed_point(center)  # Start with null point P(center)

    r_sq = radius.squeeze(-1) ** 2 if radius.dim() > center.dim() - 1 else radius ** 2

    if fix_null:
        # S = P(c) - (r^2/2) inf,  inf = e- - e+
        mv[..., 4] = mv[..., 4] + r_sq / 2
        mv[..., 5] = mv[..., 5] - r_sq / 2
    else:
        # S = P(c) - (r^2/2) (e+ + e-) = P(c) - r^2 o, the original mistake
        mv[..., 4] = mv[..., 4] - r_sq / 2
        mv[..., 5] = mv[..., 5] - r_sq / 2

    return mv
