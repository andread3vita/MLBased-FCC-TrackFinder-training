"""What each algebra's drift-hit encoding actually knows about the hit.

Three questions, answered against real hits rather than by inspection:

1. Is our wire direction the wire direction? It was built as
   (sin s cos a, sin s sin a, cos s), which tilts a stereo wire radially.
   The detector tilts it azimuthally. --fix_wire_dir switches to the
   detector's (sin s sin a, -sin s cos a, cos s).

2. Does the corrected constraint plane contain the true tangency points?
   Checked through the CGA inner product, not just Euclidean dot products,
   so a sign or convention error in embed_plane would show up too.

3. What does GGTF's two-point encoding keep, and what does it drop? The
   pair (left, right) is w +- r * x', where x' is a convention axis built
   from wire geometry alone. So w and r survive it and the wire direction
   does not.

Usage:
    python -m src.eval.audit_drift_encoding --seeds 1,2,3
"""

import argparse

import numpy as np
import polars as pl
import torch

DATA = "/home/marko.cechovic/cgatr-data/data-final/parquet"


def wire_dir_old(azim, stereo):
    """The formula every checkpoint so far trained on."""
    return np.stack([np.sin(stereo) * np.cos(azim),
                     np.sin(stereo) * np.sin(azim),
                     np.cos(stereo)], -1)


def wire_dir_new(azim, stereo):
    """The detector convention that produced left/right in the parquet."""
    return np.stack([np.sin(stereo) * np.sin(azim),
                     -np.sin(stereo) * np.cos(azim),
                     np.cos(stereo)], -1)


def load(seeds):
    frames = []
    for s in seeds:
        frames.append(pl.read_parquet(f"{DATA}/seed_{s}/dc_hits_train.parquet"))
    dc = pl.concat(frames)
    g = lambda c: dc[c].to_numpy()
    W = np.stack([g("wire_x"), g("wire_y"), g("wire_z")], 1)
    L = np.stack([g("left_x"), g("left_y"), g("left_z")], 1)
    R = np.stack([g("right_x"), g("right_y"), g("right_z")], 1)
    a, s, r = g("wire_azimuthal_angle"), g("wire_stereo_angle"), g("drift_distance")
    # A zero-drift hit has left == right and so no drift axis at all.
    keep = np.linalg.norm(R - L, axis=1) > 1e-9
    return W[keep], L[keep], R[keep], a[keep], s[keep], r[keep], (~keep).sum()


def q1_wire_direction(W, L, R, a, s):
    print("1. Is our wire direction the wire direction?")
    u = R - L
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    new = wire_dir_new(a, s)
    old = wire_dir_old(a, s)

    # The drift axis is perpendicular to the wire by construction, so a correct
    # wire direction must be orthogonal to it for every hit.
    print("   perpendicular to the drift axis the parquet carries?")
    for name, v in (("detector (--fix_wire_dir)", new), ("ours (old default)", old)):
        d = np.abs((v * u).sum(1))
        print(f"     {name:26s} |dot| mean={d.mean():.2e}  p99={np.percentile(d, 99):.2e}  max={d.max():.2e}")

    ang = np.degrees(np.arccos(np.clip(np.abs((old * new).sum(1)), 0, 1)))
    print(f"   old vs detector: median {np.median(ang):.2f} deg, p99 {np.percentile(ang, 99):.2f}, max {ang.max():.2f}")

    # A stereo wire is tilted around the beam axis, so its transverse component
    # should lie along phi, not along the radius.
    phi = np.arctan2(W[:, 1], W[:, 0])
    rad = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], 1)
    azi = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], 1)
    print("   transverse tilt (a stereo wire tilts azimuthally):")
    for name, v in (("detector (--fix_wire_dir)", new), ("ours (old default)", old)):
        print(f"     {name:26s} |along radius|={np.abs((v * rad).sum(1)).mean():.4f}"
              f"  |along phi|={np.abs((v * azi).sum(1)).mean():.4f}")
    print()


def q2_plane_contains_tangency(W, L, R, a, s, r):
    print("2. Does the constraint plane contain the true tangency points?")
    from src.cgatr.interface.plane import embed_plane
    from src.cgatr.interface.point import embed_point
    from src.cgatr.interface.sphere import embed_sphere
    from src.cgatr.primitives.invariants import compute_inner_product_mask

    gp = torch.load("cga_utils/cga_geometric_product.pt",
                    weights_only=False).to_dense().to(torch.float64)
    ipw = compute_inner_product_mask(gp, device=torch.device("cpu")).to(torch.float64)

    t = lambda x: torch.tensor(x, dtype=torch.float64)
    P = embed_point(t(L))
    S = embed_sphere(t(W), t(r), fix_null=True)
    # A point lies on the circle iff it lies on both the drift sphere and the
    # plane; only the plane depends on the wire direction, so the sphere term
    # is the control that must stay near zero either way.
    ip = lambda x, y: ((x * y) * ipw).sum(-1)
    print("   <embed_point(left), plane>  (0 means left lies on the plane)")
    for name, f in (("detector (--fix_wire_dir)", wire_dir_new),
                    ("ours (old default)", wire_dir_old)):
        pi = embed_plane(t(f(a, s)), t(W), fix_null=True)
        v = ip(P, pi).abs().numpy()
        print(f"     {name:26s} mean={v.mean():.3e}  median={np.median(v):.3e}  max={v.max():.3e}")
    v = ip(P, S).abs().numpy()
    print(f"   control, <embed_point(left), sphere>: mean={v.mean():.3e} max={v.max():.3e}")
    print()


def q3_information(W, L, R, a, s, r):
    print("3. What survives GGTF's two-point encoding?")
    w_hat = 0.5 * (L + R)
    r_hat = 0.5 * np.linalg.norm(R - L, axis=1)
    print(f"   wire position recovered as the midpoint : max err {np.abs(w_hat - W).max():.2e} mm")
    print(f"   drift radius recovered as half the gap  : max err {np.abs(r_hat - r).max():.2e} mm")

    # x' = normalise([1, 0, -d_x/d_z]) is a function of d_x/d_z alone, so it is
    # blind to d_y: a one-parameter family of wire directions shares one axis.
    d = wire_dir_new(a, s)
    ratio = d[:, 0] / d[:, 2]
    order = np.argsort(ratio)
    rs, ds = ratio[order], d[order]
    # Walk the sorted ratios and measure how far apart the true wire directions
    # get among hits whose encoded axis is indistinguishable.
    tol = 1e-6
    worst, worst_at = 0.0, None
    i = 0
    while i < len(rs):
        j = np.searchsorted(rs, rs[i] + tol, side="right")
        if j - i > 1:
            blk = ds[i:j]
            c = np.clip(np.abs(blk @ blk[0]), 0, 1)
            spread = np.degrees(np.arccos(c.min()))
            if spread > worst:
                worst, worst_at = spread, (j - i)
        i = max(j, i + 1)
    print(f"   wire direction is NOT recovered: x' depends on d_x/d_z only, so it")
    print(f"     is blind to d_y. Hits sharing an axis to {tol:g} differ in true wire")
    print(f"     direction by up to {worst:.2f} deg (largest such group: {worst_at} hits).")
    print(f"   => the pair keeps 2 of the wire's 3 dof and 1 of its 2 direction dof;")
    print(f"      the circle's plane is left underdetermined.")
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="1,2,3")
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    W, L, R, a, s, r, ndeg = load(seeds)
    print(f"seeds {seeds}: {len(W):,} drift hits with a drift axis "
          f"({ndeg:,} zero-drift hits skipped)")
    sd = np.degrees(np.abs(s))
    print(f"stereo angle: median {np.median(sd):.2f} deg, max {sd.max():.2f}\n")

    q1_wire_direction(W, L, R, a, s)
    q2_plane_contains_tangency(W, L, R, a, s, r)
    q3_information(W, L, R, a, s, r)


if __name__ == "__main__":
    main()
