"""Merge candidate fragments that lie on one circle in the transverse plane.

The clone population is dominated by curler fragmentation, and the existing
merge criterion cannot target it. That one fuses candidates whose condensation
points sit close together in the embedding, which is a statement about what the
network believes, so where the network has split a curler into arcs it will not
put them back. It mostly picks up debris instead, and under the clone-sensitive
metric that made things worse rather than better.

Geometry is available and truth-free. In a solenoid a charged track is a helix,
so its projection into the transverse plane is a circle, and a curler's
successive loops are the SAME circle traversed again. Fitting a circle per
candidate and merging those that agree on centre and radius therefore targets
exactly the failure mode, with no reference to the network's embedding and no
reference to truth.

Two guards matter:

- **Radius cap.** A stiff track has a huge bending radius and its arc is nearly
  straight, so the fit is degenerate and any two straight tracks look like
  concentric giants. Only candidates below `r_max` are eligible, which is the
  curler regime anyway: 1 GeV in the 2 T IDEA field bends at 1.7 m, and the
  fragmentation problem lives far below that.
- **Fit quality.** A candidate that is not an arc at all -- debris, or a merge of
  two particles -- will fit some circle badly. Candidates whose residual is a
  large fraction of their radius are left alone.

A z window is applied on top, because two different particles can share a
transverse circle while sitting at opposite ends of the chamber.
"""

import numpy as np

# 1 GeV bends at ~1.7 m in the 2 T field; above this a fit is a straight line
# with a meaningless centre.
DEFAULT_R_MAX = 900.0      # mm
DEFAULT_TOL = 0.25         # fraction of the smaller radius
DEFAULT_RESID = 0.15       # fit residual, fraction of the radius
DEFAULT_Z_GAP = 600.0      # mm of allowed gap between the two z ranges
DEFAULT_MIN_HITS = 10


def fit_circle(x, y):
    """Algebraic circle fit. Returns (cx, cy, r, mean_abs_residual).

    Kasa's linearisation: with x^2 + y^2 = 2 a x + 2 b y + c the normal
    equations are linear in (a, b, c), the centre is (a, b) and the radius is
    sqrt(c + a^2 + b^2). Biased for short arcs, which is acceptable here because
    the decision is a tolerance on agreement between two fits of the same kind,
    not an unbiased radius measurement.
    """
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan, np.inf
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones(n)])
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.inf
    cx, cy, c = sol
    disc = c + cx * cx + cy * cy
    if not np.isfinite(disc) or disc <= 0:
        return np.nan, np.nan, np.nan, np.inf
    r = float(np.sqrt(disc))
    resid = float(np.mean(np.abs(np.hypot(x - cx, y - cy) - r)))
    return float(cx), float(cy), r, resid


def merge_helix(labels, pos, betas, min_hits=DEFAULT_MIN_HITS,
                r_max=DEFAULT_R_MAX, tol=DEFAULT_TOL, resid_tol=DEFAULT_RESID,
                z_gap=DEFAULT_Z_GAP):
    """Relabel candidates that share a transverse circle onto one label.

    `pos` is the per-hit detector position in mm, aligned with `labels`. Only
    candidates with at least `min_hits` hits are considered, since a fit through
    a handful of points says nothing. The surviving label of a merged group is
    the one whose seed hit has the highest beta, matching what merge_fragments
    does, so downstream code that treats the label as a seed index still works.
    """
    uniq = np.unique(labels[labels >= 0])
    if len(uniq) < 2:
        return labels

    fits = {}
    for c in uniq:
        m = labels == c
        if int(m.sum()) < min_hits:
            continue
        p = pos[m]
        cx, cy, r, resid = fit_circle(p[:, 0], p[:, 1])
        if not np.isfinite(r) or r > r_max or r <= 0:
            continue
        if resid > resid_tol * r:
            continue
        fits[int(c)] = (cx, cy, r, p[:, 2].min(), p[:, 2].max())

    keys = sorted(fits)
    if len(keys) < 2:
        return labels

    parent = {k: k for k in keys}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(keys):
        ax, ay, ar, az0, az1 = fits[a]
        for b in keys[i + 1:]:
            bx, by, br, bz0, bz1 = fits[b]
            scale = min(ar, br)
            if abs(ar - br) > tol * scale:
                continue
            if np.hypot(ax - bx, ay - by) > tol * scale:
                continue
            # z ranges must touch or nearly touch
            gap = max(az0 - bz1, bz0 - az1, 0.0)
            if gap > z_gap:
                continue
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

    groups = {}
    for k in keys:
        groups.setdefault(find(k), []).append(k)

    out = labels.copy()
    for members in groups.values():
        if len(members) < 2:
            continue
        # the label whose seed hit carries the highest beta becomes the root
        root = max(members, key=lambda c: float(betas[c]) if c < len(betas) else 0.0)
        for c in members:
            if c != root:
                out[labels == c] = root
    return out


def helix_merge_stats(labels_before, labels_after):
    """How many candidates the pass removed, for logging."""
    nb = len(np.unique(labels_before[labels_before >= 0]))
    na = len(np.unique(labels_after[labels_after >= 0]))
    return nb, na
