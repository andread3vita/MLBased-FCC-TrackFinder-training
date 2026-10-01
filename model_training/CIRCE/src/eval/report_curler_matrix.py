"""Report the curler 2x2: train with/without loopers, evaluate with/without.

Reads whatever cells `run_curler_matrix.sh` and `launch_phase1_loopers.sh` have
produced and prints them together. Missing cells are shown as gaps rather than
silently dropped, since the interesting comparison is between cells and a half
matrix is easy to misread.

Read GGTF>10 as a within-table comparison. **No cell here is comparable to their
published 8%**, including the train-without row, and the earlier claim in this
docstring that Phase 1 would supply such a cell was wrong.

The reason is that the filter this table is built on is ours, not theirs. Their
loader reaches the extent cut only through `if remove_lowEnergyParticle:`, and that
flag is hardcoded False in the branch their config selects, so it never runs; what
does run relabels sub-3-hit and secondary particles as noise while leaving every hit
in the graph for the network to see. Deleting hits, as `--drop_loopers` does, is a
different operation on a different target set. See
paper_adjustment_candidates/curler_question.md, finding G1.

So the train-without row is our own ablation on looper sensitivity, which is a
perfectly good thing to measure and is not a reproduction of anything. A cell that
could face their published figure would have to reproduce
`create_garbage_label(minNumHits=3)` with relabelling instead of deletion, and no
such cell exists yet.

GGTF>10 can exceed 100% without anything being wrong: the assignment is one-to-one,
so a particle split into six candidates contributes one match and five fakes. The
clone/spurious split below each table says which of the two is happening.

    PYTHONPATH=. python src/eval/report_curler_matrix.py
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

BASE = "/home/marko.cechovic/cgatr-paper/literature/cgatr_fcc_pkg/eval_results"
GGTF_PUBLISHED = 8.0  # per cent, IDEA slide, N_hits > 10

# Cells we have decided not to fill, so they are reported as such instead of sitting in
# the "pending" list forever and reading as work still to come.
NOT_PLANNED = {
    ("projective_fixed", "without"):
        "plain projective reaches 1.6% def2, and looper sensitivity measured on a "
        "model that does not work is noise. Its train-with row already makes the "
        "encoding point it exists for; Phase 1 trains projective_ggtf instead.",
}

REGIMES = ["with", "without"]


# Cells written before the operating point was part of the directory name.
LEGACY_OP = "op (legacy: tbeta 0.1, no hit cut)"


def load(root, op=None):
    """cells[(arm, eval_regime, helix)] = summary dict.

    Cell directories are named `<operating point>_hx<tol>`, and the operating point
    changed once already (tbeta 0.1 with no hit cut, to tbeta 0.6 with a hit cut of
    4), which for the conformal arm moved candidates per event from 834 to 256 at an
    unchanged fake rate. Mixing the two in one table would compare postprocessing
    against method, so read one operating point at a time and say which. Default to
    the newest present rather than a hardcoded name, so this survives the next
    re-tune.
    """
    found = sorted(glob.glob(f"{root}/*/*/*_hx*/summary.json"))
    ops = {os.path.basename(os.path.dirname(p)).rsplit("_hx", 1)[0] for p in found}
    # Legacy cells predate the operating point being in the path at all.
    ops = {o if o != "op" else LEGACY_OP for o in ops}
    if op is None and ops:
        op = sorted(ops)[-1]

    cells = {}
    for path in found:
        parts = path.split(os.sep)
        arm, regime, leaf = parts[-4], parts[-3], parts[-2]
        cell_op, helix = leaf.rsplit("_hx", 1)
        if cell_op == "op":
            cell_op = LEGACY_OP
        if cell_op != op:
            continue
        with open(path) as f:
            cells[(arm, regime, helix)] = json.load(f)
    return cells, op, sorted(ops)


def row(s):
    return (
        s["n_events"],
        s["candidates_per_event"],
        100 * s["idea"]["match_rate"],
        100 * s["idea"]["def2"],
        100 * s["ghost_rate"],
        100 * s["clone_rate"],
        100 * s.get("ggtf_fake_rate_gt10", float("nan")),
        clone_share(s, 10),
    )


def clone_share(s, cut):
    """Percentage of the fakes above `cut` that are clones rather than spurious.

    This replaces the old GGTF>3 column, which carried no information at our
    operating point: min_cluster_hits is 4, so `cluster_size > 3` is true of every
    surviving candidate and GGTF>3 was identically GGTF>0. Older cells predate the
    split and report nan.
    """
    c = s.get(f"ggtf_n_fake_clone_gt{cut}")
    sp = s.get(f"ggtf_n_fake_spurious_gt{cut}")
    if c is None or sp is None or (c + sp) == 0:
        return float("nan")
    return 100 * c / (c + sp)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=f"{BASE}/curler_matrix")
    ap.add_argument("--phase1_root", default=f"{BASE}/curler_matrix_trained_without")
    ap.add_argument("--op", default=None,
                    help="operating point to report, as it appears in the cell "
                         "directory names. Defaults to the newest one present. "
                         "Cells from different operating points are never mixed.")
    args = ap.parse_args()

    roots = [(r, t) for r, t in ((args.root, "with"), (args.phase1_root, "without"))
             if os.path.isdir(r)]

    # Resolve the operating point once across both roots. Doing it per-root would let
    # the train-with and train-without halves of the same table come from different
    # operating points as soon as one half is re-tuned before the other.
    ops_seen = set()
    for root, _ in roots:
        ops_seen |= set(load(root)[2])
    # The legacy cells are only ever the fallback: they are the pre-re-tune numbers
    # and are kept on disk for reference, not to be reported by default.
    op_used = args.op or (sorted(ops_seen, key=lambda o: (o != LEGACY_OP, o))[-1]
                          if ops_seen else None)

    cells = {}
    for root, train in roots:
        for (arm, ev, hx), s in load(root, op_used)[0].items():
            cells[(arm, train, ev, hx)] = s

    if not cells:
        raise SystemExit(f"no cells under {args.root} or {args.phase1_root}"
                         + (f" at operating point {args.op}" if args.op else ""))

    arms = sorted({k[0] for k in cells})
    helices = sorted({k[3] for k in cells})

    print(__doc__)
    print(f"operating point: {op_used}")
    others = [o for o in sorted(ops_seen) if o != op_used]
    if others:
        print(f"also on disk, not shown: {', '.join(others)}  (--op to select)")
    for hx in helices:
        label = "no helix merge" if hx in ("0", "0.0") else f"helix merge tol {hx}"
        print(f"\n{'=' * 104}\n{label}\n{'=' * 104}")
        print(f"{'arm':<18}{'train':>7}{'eval':>9}{'events':>8}{'cand/ev':>9}"
              f"{'def1':>8}{'def2':>8}{'ghost':>8}{'clone':>8}"
              f"{'GGTF>10':>9}{'clone%':>8}")
        print("-" * 104)
        for arm in arms:
            for train in REGIMES:
                for ev in REGIMES:
                    s = cells.get((arm, train, ev, hx))
                    if s is None:
                        continue
                    (n, cpe, d1, d2, gh, cl, g10, clone_pc) = row(s)
                    print(f"{arm:<18}{train:>7}{ev:>9}{n:>8}{cpe:9.1f}"
                          f"{d1:7.1f}%{d2:7.1f}%{gh:7.1f}%{cl:7.1f}%"
                          f"{g10:8.1f}%{clone_pc:7.0f}%")
        print()
        for arm in arms:
            a = cells.get((arm, "with", "with", hx))
            b = cells.get((arm, "with", "without", hx))
            if a and b:
                # Deliberately not printed next to GGTF_PUBLISHED. Both of these
                # cells are trained on unfiltered events, so neither is the
                # configuration their 8% was measured in, and putting the numbers
                # side by side reads as a 20x deficit that the comparison does not
                # support. What the pair does show is the denominator effect: the
                # filter removes targets faster than candidates.
                mt_a = a.get("ggtf_n_matched_gt10", 0)
                mt_b = b.get("ggtf_n_matched_gt10", 0)
                print(f"  {arm}: removing loopers at EVAL time moves GGTF>10 "
                      f"from {100 * a.get('ggtf_fake_rate_gt10', float('nan')):.1f}% "
                      f"to {100 * b.get('ggtf_fake_rate_gt10', float('nan')):.1f}%, "
                      f"on a denominator falling {mt_a} -> {mt_b} matched "
                      f"while candidates only fall "
                      f"{a.get('ggtf_n_cand_gt10', 0)} -> {b.get('ggtf_n_cand_gt10', 0)}")
            c = cells.get((arm, "without", "with", hx))
            d = cells.get((arm, "without", "without", hx))
            if c and d:
                print(f"  {arm}: a model TRAINED without loopers scores "
                      f"def1 {100 * d['idea']['match_rate']:.1f}% on the filtered "
                      f"target set and {100 * c['idea']['match_rate']:.1f}% on an "
                      f"unfiltered detector")
                # Deliberately not printed next to GGTF_PUBLISHED either. This cell
                # is filtered on both sides, which used to be the argument for
                # comparing it to their figure, but the filter is ours -- their
                # extent cut never executes. What the cell does measure is whether
                # a network that never saw a long track still fragments one.
                print(f"  {arm}: GGTF>10 {100 * d.get('ggtf_fake_rate_gt10', float('nan')):.1f}% "
                      f"trained AND evaluated on the filtered set, "
                      f"{clone_share(d, 10):.0f}% of the fakes being clones; this is "
                      f"our looper ablation and not comparable to their published "
                      f"{GGTF_PUBLISHED:.0f}%")

    missing = [(arm, t, e, hx) for arm in arms for t in REGIMES for e in REGIMES
               for hx in helices
               if (arm, t, e, hx) not in cells and (arm, t) not in NOT_PLANNED]
    if missing:
        print(f"\n{len(missing)} cells still pending, e.g. {missing[:4]}")
    for (arm, t), why in sorted(NOT_PLANNED.items()):
        print(f"\n{arm}, train-{t}: not planned -- {why}")


if __name__ == "__main__":
    main()
