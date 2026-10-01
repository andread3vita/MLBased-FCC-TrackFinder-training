"""Check the projective distance-aware attention features against the reference.

GATr's distance-aware attention (Brehmer et al. arXiv:2305.18415 App. B, and
`gatr/primitives/attention.py` in the Qualcomm release) is supposed to make the
attention logit equal to the negative squared Euclidean distance between the
two points a pair of trivectors represents. That is a projective statement: it
must not depend on the homogeneous weights, which are gauge.

This script feeds the same two points to the reference construction and to
ours, with deliberately different homogeneous weights, and reports whether each
recovers -|x_q - x_k|^2.
"""

import numpy as np
import torch

from src.cgatr.primitives.attention import pga_distance_features, lin_square_normalizer


def reference_features(tri, query):
    """Qualcomm's construction: normalise the trivector, then go quadratic.

    tri is ordered (p1, p2, p3, w) -- homogeneous weight last, as in the
    reference `_TRIVECTOR_IDX = [11, 12, 13, 14]`.
    """
    r3 = torch.arange(3)
    basis = torch.zeros((4, 4, 5), dtype=tri.dtype)
    if query:
        basis[r3, r3, 0] = 1
        basis[3, 3, 1] = 1
        basis[r3, 3, 2 + r3] = 1
    else:
        basis[3, 3, 0] = -1
        basis[r3, r3, 1] = -1
        basis[r3, 3, 2 + r3] = 2
    tri_normed = tri * lin_square_normalizer(tri[..., [3]])
    return torch.einsum("xyz,...x,...y->...z", basis, tri_normed, tri_normed)


def make_point(x, weight):
    """Trivector for Euclidean point x at homogeneous weight `weight`.

    Returns both orderings: reference (weight last) and ours (weight first).
    """
    ref = torch.tensor([*(weight * x), weight], dtype=torch.float64)
    ours = torch.tensor([weight, *(weight * x)], dtype=torch.float64)
    return ref, ours


def main():
    torch.manual_seed(0)
    xq = np.array([0.30, -0.70, 1.10])
    xk = np.array([-0.40, 0.20, 0.50])
    truth = -float(((xq - xk) ** 2).sum())

    print(f"points  q = {xq},  k = {xk}")
    print(f"target logit  -|xq - xk|^2 = {truth:+.6f}\n")
    print(f"{'w_q':>6} {'w_k':>6} {'reference':>14} {'ours':>14} {'ours/target':>12}")
    print("-" * 56)

    rows = []
    for wq, wk in [(1.0, 1.0), (2.0, 1.0), (2.0, 3.0), (5.0, 0.5), (10.0, 10.0)]:
        ref_q, our_q = make_point(xq, wq)
        ref_k, our_k = make_point(xk, wk)

        r = float(
            (reference_features(ref_q, True) * reference_features(ref_k, False)).sum()
        )
        o = float(
            (
                pga_distance_features(our_q, lin_square_normalizer, query=True)
                * pga_distance_features(our_k, lin_square_normalizer, query=False)
            ).sum()
        )
        rows.append((wq, wk, r, o))
        print(f"{wq:6.1f} {wk:6.1f} {r:14.6f} {o:14.6f} {o / truth:12.3f}")

    print()
    ref_ok = all(abs(r - truth) < 2e-2 for *_, r, _ in rows)
    our_ok = all(abs(o - truth) < 2e-2 for *_, o in rows)
    print(f"reference recovers the squared distance for every gauge: {ref_ok}")
    print(f"ours      recovers the squared distance for every gauge: {our_ok}")

    if not our_ok:
        ratios = [o / truth for *_, o in rows]
        weights = [wq * wk for wq, wk, _, _ in rows]
        print("\nours / target tracks w_q * w_k:")
        for (wq, wk, _, _), ratio in zip(rows, ratios):
            print(f"  w_q*w_k = {wq * wk:7.2f}   ratio = {ratio:7.3f}")
        corr = np.corrcoef(weights, ratios)[0, 1]
        print(f"  correlation = {corr:.6f}")


if __name__ == "__main__":
    main()
