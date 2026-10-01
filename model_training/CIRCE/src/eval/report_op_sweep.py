"""Each arm at its own operating point, under GGTF's target rule and their fake definition.

The point that produced our published numbers, `tbeta 0.6 / td 0.2`, was chosen on one arm. `td`
is a radius in an embedding whose scale each arm learns for itself, so transplanting it is not
neutral: the arm it was not chosen for pays in clones, which GGTF's one-to-one assignment charges
for. This report picks each arm's own best point so the remaining difference is the model.

Selection is on GGTF>10, their headline cut, with efficiency reported beside it because a point
that suppresses every candidate would win on fakes alone. A point is only interesting if it does
not cost efficiency, so the table shows both and the choice is made on fakes at no more than 2pt
of def2 below that arm's best def2.

    PYTHONPATH=. python src/eval/report_op_sweep.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os

DEF2_TOLERANCE = 2.0   # points of def2 we will give up to buy a better fake rate


def collect(root: str, arm: str, n_tag: str) -> list[dict]:
    rows = []
    for d in sorted(glob.glob(os.path.join(root, arm, f"tb*_td*_n{n_tag}"))):
        p = os.path.join(d, "summary.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        nc = s.get("no_cuts", {})
        idea = s.get("idea", {})
        m10 = s.get("ggtf_n_matched_gt10", 0)
        n_cand10 = s.get("ggtf_n_cand_gt10", 0)
        n_fake10 = (
            s.get("ggtf_n_fake_clone_gt10", 0)
            + s.get("ggtf_n_fake_spurious_gt10", 0)
        )
        rows.append(dict(
            tag=os.path.basename(d),
            tbeta=s.get("tbeta"), td=s.get("td"),
            n_ev=s.get("n_events", 0),
            targets=s.get("n_tracks_total", 0),
            cand_per_ev=s.get("candidates_per_event", float("nan")),
            eff=100 * nc.get("efficiency", float("nan")),
            def1_no_cuts=100 * nc.get("match_rate", float("nan")),
            def2=100 * nc.get("def2", float("nan")),
            def2_no_cuts=100 * nc.get("def2", float("nan")),
            def1_idea=100 * idea.get("match_rate", float("nan")),
            def2_idea=100 * idea.get("def2", float("nan")),
            idea_hit_eff=100 * idea.get("efficiency", float("nan")),
            ggtf10=100 * s.get("ggtf_fake_rate_gt10", float("nan")),
            ggtf10_unassigned_per_matched=(
                100 * s.get("ggtf_fake_rate_gt10", float("nan"))
            ),
            candidate_fake10=100 * s.get(
                "ggtf_candidate_fake_fraction_gt10",
                n_fake10 / max(n_cand10, 1),
            ),
            candidates10=n_cand10,
            matched10=m10,
            clone10=(
                100 * s.get("ggtf_n_fake_clone_gt10", 0) / m10
                if m10 else float("inf")
            ),
            spur10=(
                100 * s.get("ggtf_n_fake_spurious_gt10", 0) / m10
                if m10 else float("inf")
            ),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="eval_results/op_sweep_per_arm")
    ap.add_argument("--arms", default="projective_ggtf v2_a_nullfix")
    ap.add_argument("--n", default="400",
                    help="which sweep to report: the event cap in the directory name. "
                         "0 is the full sample. Mixing the two would put a subset row and a "
                         "full-sample row of the same point side by side as if they were "
                         "different points.")
    ap.add_argument("--json-out", default=None,
                    help="Write the validation-selected operating point per arm "
                         "for a locked test-set evaluation.")
    args = ap.parse_args()

    print(__doc__)
    print(f"Sample: {'all 1000 events' if args.n == '0' else args.n + ' events'}.")
    best = {}
    for arm in args.arms.split():
        rows = collect(args.root, arm, args.n)
        if not rows:
            print(f"  {arm}: nothing swept yet")
            continue
        print(f"\n=== {arm} ===  {rows[0]['n_ev']} events, "
              f"{rows[0]['targets'] / max(rows[0]['n_ev'], 1):.1f} targets/event")
        hdr = (
            f"{'tbeta':>6}{'td':>6}{'cand/ev':>9}{'eff all':>9}"
            f"{'def2 all':>10}{'def2 IDEA':>11}{'GGTF U/M':>10}"
            f"{'cand fake':>11}{'clone':>8}{'spurious':>10}"
        )
        print(hdr)
        print("-" * len(hdr))
        valid = [r for r in rows if r["matched10"] > 0]
        if not valid:
            print("  no point has a matched >10-hit candidate")
            continue
        best_def2 = max(r["def2"] for r in valid)
        eligible = [
            r for r in valid if r["def2"] >= best_def2 - DEF2_TOLERANCE
        ]
        pick = min(eligible, key=lambda r: r["ggtf10"])
        for r in rows:
            mark = "  <-- chosen" if r is pick else ""
            print(
                f"{r['tbeta']:6.2f}{r['td']:6.2f}{r['cand_per_ev']:9.1f}"
                f"{r['eff']:8.1f}%{r['def2_no_cuts']:9.1f}%"
                f"{r['def2_idea']:10.1f}%{r['ggtf10']:9.1f}%"
                f"{r['candidate_fake10']:10.1f}%"
                f"{r['clone10']:7.1f}%{r['spur10']:9.1f}%{mark}"
            )
        best[arm] = pick

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump({
                "selection_sample": args.n,
                "criterion": {
                    "scope": "all-target/no-cuts",
                    "primary": (
                        "minimum GGTF>10 unassigned/matched rate among points "
                        f"within {DEF2_TOLERANCE:g}pt of the arm's best def2"
                    ),
                },
                "arms": best,
            }, f, indent=2)
        print(f"\nLocked operating points written to {args.json_out}")

    if len(best) < 2:
        return
    print("\n=== each arm at its own best point ===")
    hdr = f"{'arm':<22}{'tbeta':>7}{'td':>6}{'eff':>8}{'def2':>8}{'GGTF>10':>9}{'clone':>8}"
    print(hdr)
    print("-" * len(hdr))
    for arm, r in best.items():
        print(f"{arm:<22}{r['tbeta']:7.2f}{r['td']:6.2f}{r['eff']:7.1f}%"
              f"{r['def2']:7.1f}%{r['ggtf10']:8.1f}%{r['clone10']:7.1f}%")

    arms = list(best)
    if len(arms) == 2:
        a, b = best[arms[0]], best[arms[1]]
        print(f"\nGap at own-best points: {b['ggtf10'] - a['ggtf10']:+.1f}pt of GGTF>10 and "
              f"{b['def2'] - a['def2']:+.1f}pt of def2 for {arms[1]} against {arms[0]}.")
        print("Compare that against the gap at the single transplanted point before concluding")
        print("anything about the algebras. If the gap shrinks a lot, the transplanted point was")
        print("doing the work; if it barely moves, the difference is the model.")
    print("\nSubset runs rank points; confirm the chosen pair with FULL=1 before quoting.")


if __name__ == "__main__":
    main()
