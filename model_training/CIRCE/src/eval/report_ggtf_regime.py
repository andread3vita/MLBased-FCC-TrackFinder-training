"""Our arms scored the way GGTF actually score, now that we have read their code.

Four things separate this from the numbers in the five-arm table, and only the first was
already right there:

1. **Unfiltered input.** Their `remove_lowEnergyParticle` sits behind a hardcoded False and
   never runs, and `remove_loopers_overlay` is reachable only with overlay on (G1, G2). So
   every hit is in the event. Our published table already used this regime.
2. **Their target rule.** `create_garbage_label(minNumHits=3)` relabels a particle with fewer
   than three hits to noise while keeping its hits in the graph (G3). We had been counting
   every particle with at least one hit as a target, which is 143.8 per event against their
   ~32. This is the big one: it changes the denominator by a factor of four.
3. **Their fake definition.** Unassigned over *matched* candidates under a one-to-one
   assignment with an IoU >= 0.02 gate (G7), which is unbounded above and charges for clones.
4. **Particle 0 counted.** It is a real particle that our pipeline called noise (M20). Only
   available where the cache was regenerated with FIX_PARTICLE_ZERO=1, since older caches do
   not contain those hits at all (M22).

What is still *not* comparable to their published 8%, and would be dishonest to present as
though it were: their number comes from their own network on their own sample, where a track
carries a median 22 hits against our 117 (M16), and the two open questions in the mail to
Andrea and Dolores -- which side their `> 3 unique hits` cut applies to, and whether
`remove_lowEnergyParticle` was on for the published run -- move our figure by tens of points.
This table says where we stand under their definitions on our data. It does not close the gap.

    PYTHONPATH=. python src/eval/report_ggtf_regime.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import polars as pl

from src.eval.ggtf_fake_rate import add_iou, ggtf_rates

# Their quote for an IDEA event, from the same slide as the 8%.
THEIR_TARGETS_PER_EVENT = 32

# Ordered so the table reads as an argument rather than a pile: the four parity cells first,
# because they are the only ones trained under GGTF's own target rule and so the only ones
# whose numbers can be set beside theirs (M34); then the pre-parity arms that the parity cells
# are measured against; then the historical arms, which are all bugged one way or another and
# are kept only so the progression is legible.
#
# `v2_pzero` sits directly under `conformal_parity` on purpose: they differ by exactly one
# training flag, `--min_target_hits 3`, so the pair is the measurement of what GGTF's target
# rule costs or buys (M33).
ARMS = [
    ("conformal_parity", "conformal @ parity"),
    ("projective_parity", "projective @ parity"),
    ("conformal_parity_time", "conformal @ parity + time"),
    ("projective_parity_time", "projective @ parity + time"),
    ("v2_pzero", "conformal + null + pzero (pre)"),
    ("projective_ggtf", "projective + their encoding"),
    ("v2_a_nullfix", "conformal + null fix"),
    ("v2_b_time", "conformal + null fix + time"),
    ("conformal_time", "conformal + hit time"),
    ("conformal_asis", "conformal, legacy equivariance"),
    ("conformal", "conformal"),
    ("projective_fixed", "projective, our encoding"),
]


def load(scored_dir: str):
    cl_path = os.path.join(scored_dir, "cache_clusters.parquet")
    tr_path = os.path.join(scored_dir, "cache.parquet")
    if not (os.path.exists(cl_path) and os.path.exists(tr_path)):
        return None
    cols = ["cluster_size", "matched_mc_idx", "best_match", "purity",
            "event_id", "seed", "is_fake_idea"]
    have = pl.scan_parquet(cl_path).collect_schema().names()
    if "ggtf_assigned" in have:
        cols.append("ggtf_assigned")
    clusters = pl.read_parquet(cl_path, columns=cols)
    tracks = pl.read_parquet(tr_path, columns=[
        "mc_idx", "event_id", "seed", "n_hits_signal", "gen_status",
        "vertex_r", "theta_deg"])
    return add_iou(clusters, tracks), tracks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="eval_results/ggtf_regime",
                    help="dir holding <arm>/<op> scored outputs")
    ap.add_argument("--op", default="tb0.6_td0.2_mh4_hmh4_hx0.15_mth3")
    args = ap.parse_args()

    print(__doc__)
    rows = []
    for arm, label in ARMS:
        scored = os.path.join(args.root, arm, args.op)
        got = load(scored)
        if got is None:
            print(f"  {arm}: not scored yet ({scored})")
            continue
        clusters, tracks = got
        summary_path = os.path.join(scored, "summary.json")
        s = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
        n_ev = s.get("n_events", tracks.select(["seed", "event_id"]).unique().height)
        plain = ggtf_rates(clusters, tracks, n_events=n_ev)
        phys = ggtf_rates(clusters, tracks, physics_cuts=True, n_events=n_ev)
        # What the same arm and the same cache scored before the target rule was corrected,
        # so the table shows what the correction did rather than only where we land.
        old_path = os.path.join(
            "eval_results/curler_matrix", arm, "with",
            args.op.replace("_mth3", ""), "summary.json")
        old = json.load(open(old_path)) if os.path.exists(old_path) else {}

        nc = s.get("no_cuts", {})
        m10 = max(s.get("ggtf_n_matched_gt10", 0), 1)
        rows.append(dict(
            arm=arm, label=label, n_ev=n_ev,
            targets_per_ev=s.get("n_tracks_total", 0) / max(n_ev, 1),
            def2=100 * nc.get("def2", float("nan")),
            eff=100 * nc.get("efficiency", float("nan")),
            ggtf3=plain.get("ggtf>3"), ggtf10=plain.get("ggtf>10"),
            ggtf3p=phys.get("ggtf>3"), ggtf10p=phys.get("ggtf>10"),
            clone10=100 * s.get("ggtf_n_fake_clone_gt10", 0) / m10,
            spur10=100 * s.get("ggtf_n_fake_spurious_gt10", 0) / m10,
            old10=100 * old["ggtf_fake_rate_gt10"] if old else None,
            old_targets=old.get("n_tracks_total", 0) / max(n_ev, 1) if old else None,
            fix_pz=s.get("fix_particle_zero", False),
            mth=s.get("min_target_hits", 0),
        ))

    if not rows:
        print("\nNothing scored. Run run_ggtf_regime.sh first.")
        return

    print(f"\nOperating point {args.op}, {rows[0]['n_ev']} unseen events (seeds 191-192).")
    print("GGTF's own metric: unassigned / matched candidates, one-to-one, IoU >= 0.02.")
    print("'their cuts' additionally restricts the particle side to make_cuts_ct.\n")
    hdr = (f"{'arm':<32}{'targets/ev':>11}{'eff':>7}{'def2':>7}"
           f"{'GGTF>3':>9}{'GGTF>10':>9}{'of which clone':>16}{'was':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        was = f"{r['old10']:.1f}%" if r["old10"] is not None else "n/a"
        print(f"{r['label']:<32}{r['targets_per_ev']:11.1f}"
              f"{r['eff']:6.1f}%{r['def2']:6.1f}%"
              f"{r['ggtf3']:8.1f}%{r['ggtf10']:8.1f}%"
              f"{r['clone10']:15.1f}%{was:>8}")
    print("\n'was' is the same arm and the same unfiltered cache scored with our old target")
    # Only arms whose cache predates the target-rule change carry an old-target count, and which
    # arm sorts first is not stable, so take the first row that actually has one.
    old_t = next((r["old_targets"] for r in rows if r.get("old_targets") is not None), None)
    if old_t is not None:
        print(f"rule, every particle with at least one hit, i.e. {old_t:.0f} targets per event.")
    else:
        print("rule, every particle with at least one hit.")
    print("Efficiency and def2 are over the corrected target set; def2 is the strict")
    print("definition (matched, unsplit, unshared).")

    print("\nUnder their own notebook cuts on the particle side (make_cuts_ct), which shrink")
    print("the target set further while leaving every candidate in the numerator:")
    for r in rows:
        print(f"  {r['label']:<32}>3 {r['ggtf3p']:7.1f}%   >10 {r['ggtf10p']:7.1f}%")
    print("  That combination is an upper bound rather than their number: their notebooks")
    print("  scope the efficiency denominator with these cuts, not the fake formula.")

    r0 = rows[0]
    print(f"\nTarget density is the check that the regime is right: {r0['targets_per_ev']:.1f} "
          f"per event against their quoted ~{THEIR_TARGETS_PER_EVENT}.")
    if r0["mth"] < 2:
        print("WARNING: min_target_hits is not set, so this is NOT their target rule and the")
        print("         density above should be ~144. Re-run with --min_target_hits 3.")
    if not r0["fix_pz"]:
        print("Note: particle 0 is still excluded (M20). It is ~0.2 targets per event, so it")
        print("      moves these numbers by well under a point; regenerate the caches with")
        print("      FIX_PARTICLE_ZERO=1 to include it.")
    print("\nRead the docstring above before quoting any of this against their 8%.")


if __name__ == "__main__":
    main()
