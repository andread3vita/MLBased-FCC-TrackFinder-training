"""Generate Cayley tables for the projective geometric algebra Cl(3,0,1).

This is the algebra used by GATr and by the GGTF tracker: basis vectors
e0 (e0^2 = 0), e1, e2, e3 (square to +1), 16 blades, grades 0-4 with dimensions
(1, 4, 6, 4, 1). Points are trivectors, lines bivectors, planes vectors.

The products are built directly from blade bitmaps rather than via `clifford`,
which cannot be imported in this environment (it pulls in numba, which conflicts
with the installed numpy). The construction is standard and self-verifying, and
it removes a build dependency:

    python -m src.cgatr.generate_pga_tables

Two things differ from the conformal generator and are the reason this file
exists rather than a signature argument to it:

1. The pseudoscalar is degenerate (I^2 = 0), so the dual cannot be I^{-1}
   multiplication. We use the complement dual fixed by  b ^ dual(b) = I,
   which is metric-independent and therefore well defined here.
2. Points cannot be null vectors as in the conformal algebra, so the point
   embedding is derived constructively as the meet of three planes.
"""

import os
from pathlib import Path

import numpy as np
import torch

# Basis vector order: e0 (degenerate) first, following the PGA literature.
SIG = (0, 1, 1, 1)
NV = 4
NUM_BLADES = 1 << NV
NAMES = ("e0", "e1", "e2", "e3")


def _popcount(x):
    return bin(x).count("1")


def _reordering_sign(a, b):
    """Sign from anticommuting the basis vectors of blade `a` past those of `b`."""
    a >>= 1
    total = 0
    while a:
        total += _popcount(a & b)
        a >>= 1
    return -1.0 if (total & 1) else 1.0


def _geometric(a, b):
    """Geometric product of two basis blades given as bitmaps -> (bitmap, coeff)."""
    sign = _reordering_sign(a, b)
    common = a & b
    i = 0
    while common:
        if common & 1:
            s = SIG[i]
            if s == 0:
                return 0, 0.0
            sign *= s
        common >>= 1
        i += 1
    return a ^ b, sign


def _outer(a, b):
    """Outer product of two basis blades -> (bitmap, coeff)."""
    if a & b:
        return 0, 0.0
    return a ^ b, _reordering_sign(a, b)


def _blade_order():
    """Bitmaps sorted by grade, then by bit pattern. Index 0 is the scalar."""
    bitmaps = sorted(range(NUM_BLADES), key=lambda b: (_popcount(b), b))
    index_of = {b: i for i, b in enumerate(bitmaps)}
    grades = [_popcount(b) for b in bitmaps]
    names = []
    for b in bitmaps:
        if b == 0:
            names.append("1")
        else:
            names.append("e" + "".join(NAMES[i][1] for i in range(NV) if b >> i & 1))
    return bitmaps, index_of, grades, names


def build_tables():
    bitmaps, index_of, grades, names = _blade_order()
    n = NUM_BLADES

    gp = torch.zeros(n, n, n, dtype=torch.float32)
    op = torch.zeros(n, n, n, dtype=torch.float32)
    for j, bj in enumerate(bitmaps):
        for k, bk in enumerate(bitmaps):
            r, c = _geometric(bj, bk)
            if c:
                gp[index_of[r], j, k] = c
            r, c = _outer(bj, bk)
            if c:
                op[index_of[r], j, k] = c

    # Complement dual: the unique blade with  b ^ dual(b) = +I, carrying the sign
    # that makes that identity hold. Valid in a degenerate algebra, where the
    # I^{-1} construction used for the conformal tables does not exist.
    ps = n - 1
    dual_perm = [0] * n
    dual_signs = torch.zeros(n, dtype=torch.float32)
    for j, bj in enumerate(bitmaps):
        comp = (NUM_BLADES - 1) ^ bj
        i = index_of[comp]
        _, c = _outer(bj, comp)
        assert c != 0.0, f"complement of {names[j]} vanished"
        dual_perm[j] = i
        dual_signs[j] = c

    reversal = torch.tensor(
        [(-1.0) ** (g * (g - 1) // 2) for g in grades], dtype=torch.float32
    )
    involution = torch.tensor([(-1.0) ** g for g in grades], dtype=torch.float32)

    grade_ranges, grade_dims = {}, []
    for g in range(NV + 1):
        idx = [i for i, gg in enumerate(grades) if gg == g]
        grade_ranges[g] = (idx[0], idx[-1] + 1)
        grade_dims.append(len(idx))

    meta = {
        "blade_names": names,
        "blade_bitmaps": bitmaps,
        "blade_grades": grades,
        "grade_ranges": grade_ranges,
        "dual_permutation": dual_perm,
        "dual_signs": dual_signs,
        "reversal_signs": reversal,
        "grade_involution_signs": involution,
        "signature": (3, 0, 1),
        "num_blades": n,
        "grade_dims": grade_dims,
        "pseudoscalar_index": ps,
    }
    return gp, op, meta


# ---------------------------------------------------------------------------
# Geometric embeddings, derived from the tables rather than hand-transcribed so
# that no sign convention has to be trusted.
# ---------------------------------------------------------------------------

def _mv_outer(op, x, y):
    return torch.einsum("ijk,j,k->i", op, x, y)


def point_matrix(op, meta):
    """(16, 4) matrix M with  P(x) = M @ [x, y, z, 1]  a trivector point.

    A plane with unit normal n and offset d is the vector n_x e1 + n_y e2 +
    n_z e3 + d e0. The point (X, Y, Z) is the meet of the three planes
    x = X, y = Y, z = Z, so it is the outer product of those three vectors.
    """
    idx = {nm: i for i, nm in enumerate(meta["blade_names"])}

    def plane(axis, offset):
        v = torch.zeros(NUM_BLADES)
        v[idx[f"e{axis}"]] = 1.0
        v[idx["e0"]] = offset
        return v

    cols = []
    for X, Y, Z in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)]:
        p = _mv_outer(op, _mv_outer(op, plane(1, -X), plane(2, -Y)), plane(3, -Z))
        cols.append(p)
    # The construction is affine in (X, Y, Z); subtract the origin to isolate
    # the linear part so that M @ [x, y, z, 1] reproduces it exactly.
    origin = cols[3]
    M = torch.stack([c - origin for c in cols[:3]] + [origin], dim=1)
    return M


def line_matrix(op, meta, dual_perm, dual_signs):
    """(16, 6) matrix L with  line(u, m) = L @ [u, m]  for Plucker (u, m = w x u).

    Fitted from joins of point pairs, then verified, so the bivector convention
    follows from the tables instead of being asserted.
    """
    # Double precision: the moments are O(1e4), so a float32 fit leaves a
    # residual large enough to be indistinguishable from a real error.
    opd = op.double()
    Md = point_matrix(op, meta).double()
    signs_d = dual_signs.double()

    def pt(a):
        return Md @ torch.tensor([a[0], a[1], a[2], 1.0], dtype=torch.float64)

    def join(A, B):
        d = lambda v: signs_d * v[dual_perm]
        return d(_mv_outer(opd, d(A), d(B)))

    rng = np.random.default_rng(0)
    rows_in, rows_out = [], []
    for _ in range(60):
        a = rng.normal(size=3) * 100.0
        b = rng.normal(size=3) * 100.0
        u = b - a
        m = np.cross(a, b)
        rows_in.append(np.concatenate([u, m]))
        rows_out.append(join(pt(a), pt(b)).numpy())

    A = np.asarray(rows_in)
    Y = np.asarray(rows_out)
    L, *_ = np.linalg.lstsq(A, Y, rcond=None)
    resid = np.abs(A @ L - Y).max() / np.abs(Y).max()
    assert resid < 1e-9, f"Plucker fit relative residual {resid:.2e}"
    return torch.tensor(L.T, dtype=torch.float32), float(resid)


def verify(gp, op, meta):
    idx = {nm: i for i, nm in enumerate(meta["blade_names"])}
    assert gp[0, idx["e0"], idx["e0"]] == 0.0, "e0 must square to zero"
    for a in ("e1", "e2", "e3"):
        assert gp[0, idx[a], idx[a]] == 1.0, f"{a} must square to +1"

    # Degenerate pseudoscalar: this is what rules out the I^{-1} dual.
    ps = meta["pseudoscalar_index"]
    assert gp[:, ps, ps].abs().max() == 0.0, "I^2 should vanish in Cl(3,0,1)"

    perm, signs = meta["dual_permutation"], meta["dual_signs"]
    for j in range(NUM_BLADES):
        v = torch.zeros(NUM_BLADES)
        v[j] = 1.0
        dv = torch.zeros(NUM_BLADES)
        dv[perm[j]] = signs[j]
        assert _mv_outer(op, v, dv)[ps].item() == 1.0, f"b ^ dual(b) != I for blade {j}"

    # Points must round-trip through the trivector representation.
    M = point_matrix(op, meta)
    tri = slice(*meta["grade_ranges"][3])
    rng = np.random.default_rng(1)
    for _ in range(20):
        x = rng.normal(size=3) * 500.0
        P = M @ torch.tensor([x[0], x[1], x[2], 1.0], dtype=torch.float32)
        w = P[tri]
        # The e123 coefficient is the homogeneous weight; the other three carry
        # the coordinates up to the signs fixed by the wedge construction.
        names3 = meta["blade_names"][tri]
        wt = w[names3.index("e123")]
        rec = []
        for c, nm in zip((0, 1, 2), ("e023", "e013", "e012")):
            s = -1.0 if nm in ("e023", "e012") else 1.0
            rec.append((w[names3.index(nm)] / wt * s).item())
        err = np.abs(np.array(rec) - x).max()
        assert err < 1e-2, f"point round-trip error {err:.2e} for {x}"


def main(output_dir=None):
    gp, op, meta = build_tables()
    meta["blade_names"] = list(meta["blade_names"])
    verify(gp, op, meta)

    L, resid = line_matrix(op, meta, meta["dual_permutation"], meta["dual_signs"])
    meta["point_matrix"] = point_matrix(op, meta)
    meta["line_matrix"] = L

    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent.parent / "cga_utils")
    os.makedirs(output_dir, exist_ok=True)

    torch.save(gp.to_sparse(), os.path.join(output_dir, "pga_geometric_product.pt"))
    torch.save(op.to_sparse(), os.path.join(output_dir, "pga_outer_product.pt"))
    torch.save(meta, os.path.join(output_dir, "pga_metadata.pt"))

    print(f"Saved PGA Cl(3,0,1) tables to {output_dir}")
    print(f"  blades      : {meta['num_blades']}  grades {meta['grade_dims']}")
    print(f"  gp nonzero  : {int(gp.count_nonzero())}")
    print(f"  op nonzero  : {int(op.count_nonzero())}")
    print(f"  names       : {meta['blade_names']}")
    print(f"  Plucker fit : max residual {resid:.2e}")
    print("  verified    : e0^2=0, I^2=0, b^dual(b)=I, point round-trip")
    return gp, op, meta


if __name__ == "__main__":
    main()
