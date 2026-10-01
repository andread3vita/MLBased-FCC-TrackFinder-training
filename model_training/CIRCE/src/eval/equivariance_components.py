"""Which layer breaks translation equivariance?

The end-to-end test says the backbone does not commute with a translation
versor. This walks the primitives one at a time so the failure can be pinned on
a specific layer instead of the stack as a whole.

    CUDA_VISIBLE_DEVICES= python -m src.eval.equivariance_components
"""

import torch

from src.cgatr.primitives.linear import _compute_se3_equi_linear_basis
from src.eval.smoke_arms import base_args
from src.eval.test_equivariance import rotor, sandwich, translator
from src.model import CGATrParquetModel

SHIFT = (0.31, -0.17, 0.44)
THETA = 0.7


def rel(a, b):
    return (a - b).abs().max().item() / max(a.abs().max().item(), 1e-12)


def report(label, err, tol=1e-6):
    # Loose next to the machine epsilon on purpose: a conformal translation
    # makes null-vector coefficients cancel across orders of magnitude, so an
    # exactly equivariant map still leaves a residual that grows with the shift.
    # A genuinely broken symmetry sits at O(1), far above any such tolerance.
    print(f"    [{'ok  ' if err < tol else 'FAIL'}] {label:<44} {err:.2e}")
    return err < tol


def main():
    torch.manual_seed(0)
    all_ok = True

    for arm, kw in [("conformal", {}), ("projective", {"algebra": "projective"})]:
        model = CGATrParquetModel(base_args(**kw)).eval().double()
        gp = model.basis_gp.to(torch.float64)
        nb = model.num_blades
        print(f"\n{arm}  ({nb} blades)")

        for motion, versor in [("rotation", rotor(model, THETA)),
                               ("translation", translator(model, SHIFT))]:
            print(f"  {motion}")
            x = torch.randn(24, 3, nb, dtype=torch.float64)
            y = torch.randn(24, 3, nb, dtype=torch.float64)
            xm = sandwich(model, versor, x)
            ym = sandwich(model, versor, y)

            # 1. Equivariant linear basis: every basis map must commute with the
            #    versor action, which is what "equivariant" is supposed to mean.
            basis = model.cgatr.basis_pin.to(torch.float64)
            lin = torch.einsum("aij,...j->...ai", basis, x)
            lin_moved = torch.einsum("aij,...j->...ai", basis, xm)
            all_ok &= report("equi_linear basis", rel(sandwich(model, versor, lin),
                                                      lin_moved))

            # 2. Geometric product: equivariant in any algebra, so this is a
            #    control on the Cayley table and the sandwich helper itself.
            prod = torch.einsum("ijk,...j,...k->...i", gp, x, y)
            prod_moved = torch.einsum("ijk,...j,...k->...i", gp, xm, ym)
            all_ok &= report("geometric product (control)",
                             rel(sandwich(model, versor, prod), prod_moved))

            # 3. The invariant inner product, as the attention actually computes
            #    it: a dot over a subset of blades, metric-weighted on the query
            #    side when the algebra needs it.
            attn = model.cgatr.blocks[0].attention.attention.geometric_attention
            idx = attn._INNER_PRODUCT_WO_EXTREMES_IDX
            w = (1.0 if attn.ip_weights is None
                 else attn.ip_weights.to(torch.float64))
            ip = torch.einsum("...i,...i->...", x[..., idx] * w, y[..., idx])
            ip_moved = torch.einsum("...i,...i->...", xm[..., idx] * w, ym[..., idx])
            all_ok &= report("attention inner product (as implemented)",
                             rel(ip, ip_moved))

            # 4. The true GA inner product <x~ y>_0, for comparison.
            revd = x * model._reversal.to(torch.float64)
            true_ip = torch.einsum("ijk,...j,...k->...i", gp, revd, y)[..., 0]
            revd_m = xm * model._reversal.to(torch.float64)
            true_ip_m = torch.einsum("ijk,...j,...k->...i", gp, revd_m, ym)[..., 0]
            all_ok &= report("GA inner product <x~ y>_0", rel(true_ip, true_ip_m))

            # 5. Distance-aware attention features, where present.
            if attn.use_dist:
                from src.cgatr.primitives.attention import (
                    _build_dist_vec, lin_square_normalizer)
                qd = _build_dist_vec(x[None, None, ..., attn._GRADE1_IDX],
                                     attn.basis_q.to(torch.float64),
                                     lin_square_normalizer)
                kd = _build_dist_vec(y[None, None, ..., attn._GRADE1_IDX],
                                     attn.basis_k.to(torch.float64),
                                     lin_square_normalizer)
                qdm = _build_dist_vec(xm[None, None, ..., attn._GRADE1_IDX],
                                      attn.basis_q.to(torch.float64),
                                      lin_square_normalizer)
                kdm = _build_dist_vec(ym[None, None, ..., attn._GRADE1_IDX],
                                      attn.basis_k.to(torch.float64),
                                      lin_square_normalizer)
                logit = (qd * kd).sum(-1)
                logit_m = (qdm * kdm).sum(-1)
                all_ok &= report("distance-attention logit", rel(logit, logit_m))

            if attn.use_pga_dist:
                from src.cgatr.primitives.attention import (
                    lin_square_normalizer, pga_distance_features)
                di = attn._PGA_DIST_IDX
                f = lambda t, q: pga_distance_features(  # noqa: E731
                    t[..., di], lin_square_normalizer, query=q)
                logit = (f(x, True) * f(y, False)).sum(-1)
                logit_m = (f(xm, True) * f(ym, False)).sum(-1)
                all_ok &= report("distance-attention logit (GATr App. B)",
                                 rel(logit, logit_m))

    # 6. Does the conformal linear basis change if translations are generated by
    #    the point at infinity instead of the origin null vector?
    print("\nconformal linear basis, choice of null direction")
    gp = torch.load("cga_utils/cga_geometric_product.pt",
                    weights_only=False).to_dense()
    inf = torch.zeros(32, dtype=torch.float64)
    inf[4], inf[5] = -1.0, 1.0  # e- - e+
    cur = _compute_se3_equi_linear_basis(gp, verify=False).double()
    fixed = _compute_se3_equi_linear_basis(gp, verify=False,
                                           translation_vec=inf).double()
    # Compare the spans via principal angles.
    A = cur.reshape(cur.shape[0], -1)
    B = fixed.reshape(fixed.shape[0], -1)
    A = A / A.norm(dim=1, keepdim=True)
    B = B / B.norm(dim=1, keepdim=True)
    s = torch.linalg.svdvals(A @ B.T)
    shared = int((s > 1 - 1e-6).sum())
    print(f"    e+ + e- basis: {cur.shape[0]} maps;  e- - e+ basis: "
          f"{fixed.shape[0]} maps;  shared directions: {shared}/{cur.shape[0]}")

    print("\n" + ("all components equivariant" if all_ok
                  else "at least one component is not equivariant"))


if __name__ == "__main__":
    main()
