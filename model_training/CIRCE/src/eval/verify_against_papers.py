"""Check our algebra machinery against the published Qualcomm results.

Two papers:
  GATr        Brehmer et al., arXiv:2305.18415, projective G(3,0,1)
  EPC         de Haan et al., arXiv:2311.04744, Euclidean/projective/conformal

The reference numbers, both from EPC Appendix B.2 (the projective count also
being GATr Prop. 1):

  E(3)-equivariant linear maps    PGA  9      CGA 20

Our runs are SE(3)-equivariant: EPC drops the mirror constraint and notes the
SE(3) maps are the E(3) maps "possibly combined with multiplication with the
pseudoscalar". That factor-of-two ceiling is exact for the conformal algebra and
must be smaller for the projective one, whose pseudoscalar e0123 is degenerate
and annihilates every blade containing e0. So the published numbers do not pin
the SE(3) counts down; what they pin down is the E(3) counts, and reproducing
those with the mirror constraint added is what validates the construction.

    CUDA_VISIBLE_DEVICES= python -m src.eval.verify_against_papers
"""

import torch

from src.cgatr.primitives.linear import _compute_se3_equi_linear_basis

# EPC Appendix B.2.
PUBLISHED_E3 = {"CGA": 20, "PGA": 9}


def mirror_matrix(gp, vec_idx, involution):
    """Twisted conjugation by a unit vector: x -> u x^ u^-1, a reflection.

    EPC adds exactly this as the discrete constraint on top of the six Lie
    generators to go from SE(3) to E(3).
    """
    n = gp.shape[0]
    u = torch.zeros(n, dtype=torch.float64)
    u[vec_idx] = 1.0
    # u^2 = +1 for a Euclidean basis vector, so u^-1 = u.
    left = torch.einsum("ikj,k->ij", gp, u)   # y -> u y
    right = torch.einsum("ijk,k->ij", gp, u)  # y -> y u
    return right @ left @ torch.diag(involution)


def count_e3(gp, involution, spatial_idx, translation_idx, tol=1e-6):
    """Dimension of the space of E(3)-equivariant linear maps."""
    n = gp.shape[0]
    gp64 = gp.to(torch.float64)

    se3 = _compute_se3_equi_linear_basis(
        gp, spatial_idx=spatial_idx, translation_idx=translation_idx,
        verify=False,
    ).to(torch.float64)

    # Which combinations of the SE(3) maps also commute with the mirror?
    M = mirror_matrix(gp64, spatial_idx[0], involution.to(torch.float64))
    resid = (torch.einsum("ik,bkj->bij", M, se3)
             - torch.einsum("bik,kj->bij", se3, M))
    A = resid.reshape(se3.shape[0], -1)          # (n_se3, n*n)
    s = torch.linalg.eigvalsh(A @ A.T)           # null space over coefficients
    n_e3 = int((s < tol * max(s.max().item(), 1.0)).sum())
    return se3.shape[0], n_e3


def main():
    ok = True
    print("Equivariant linear maps (ours vs EPC arXiv:2311.04744 App. B.2)\n")
    print(f"  {'algebra':<10}{'blades':>7}{'SE(3)':>8}{'E(3)':>7}{'published':>11}")

    for label, prefix, spatial, trans in [
        ("CGA", "cga", (1, 2, 3), None),
        ("PGA", "pga", (2, 3, 4), 1),
    ]:
        gp = torch.load(f"cga_utils/{prefix}_geometric_product.pt",
                        weights_only=False).to_dense()
        meta = torch.load(f"cga_utils/{prefix}_metadata.pt", weights_only=False)
        n_se3, n_e3 = count_e3(gp, meta["grade_involution_signs"], spatial, trans)
        want = PUBLISHED_E3[label]
        flag = "" if n_e3 == want else "   <-- MISMATCH"
        ok &= n_e3 == want
        print(f"  {label:<10}{gp.shape[0]:>7}{n_se3:>8}{n_e3:>7}{want:>11}{flag}")

    print("\n" + ("matches the published counts" if ok
                  else "DOES NOT MATCH — the construction is wrong"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
