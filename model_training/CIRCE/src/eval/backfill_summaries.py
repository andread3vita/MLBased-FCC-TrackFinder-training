"""Recompute summary.json for existing eval dirs without re-clustering.

fcc_cache_parallel.py grew the def2 efficiency breakdown and the ghost/clone
rates after these sweeps had already run. Clustering is the expensive step and
its output is already on disk (cache.parquet, cache_clusters.parquet), so the
new fields are recovered by re-reducing those parquets. The clustering
parameters are read back from the existing summary.json, which is then
overwritten in place; --dry_run prints without writing.

Oracle dirs are marked truth_derived and keep no fake/clone fields: their
candidates are built from truth, so a ghost rate there means nothing.

  python src/eval/backfill_summaries.py eval_results/consol_r3_nokeepall/merge_sweep/*
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import polars as pl

from src.eval.fcc_cache_parallel import candidate_stats, overall

CAND_FIELDS = ["candidates_per_event", "ghost_rate", "clone_rate", "mc_clone_rate",
               "clones_per_event", "ghosts_per_event", "fakes_per_event_idea"]


def backfill(d):
    sp = os.path.join(d, "summary.json")
    tp = os.path.join(d, "cache.parquet")
    cp = os.path.join(d, "cache_clusters.parquet")
    if not (os.path.exists(sp) and os.path.exists(tp)):
        return None
    with open(sp) as f:
        s = json.load(f)

    t = pl.read_parquet(tp, columns=["matched", "purity_of_match", "efficiency_per_hit",
                                     "is_reconstructable_idea", "is_reconstructable_cld",
                                     "is_reconstructable_displaced", "event_id", "seed"])
    n_events = len(t.select(["event_id", "seed"]).unique())
    s["n_events"] = n_events
    s["no_cuts"] = overall(t, None)
    s["idea"] = overall(t, "is_reconstructable_idea")
    s["cld"] = overall(t, "is_reconstructable_cld")
    s["displaced"] = overall(t, "is_reconstructable_displaced")

    if "oracle" in os.path.basename(d):
        s["truth_derived"] = True
        for k in CAND_FIELDS:
            s.pop(k, None)
    elif os.path.exists(cp):
        c = pl.read_parquet(cp, columns=["matched_mc_idx", "event_id", "seed",
                                         "purity", "is_fake_idea", "is_fake_cld"])
        s.update(candidate_stats(c, n_events))
        # recomputed, not carried over: the fake flags depend on the
        # reconstructable selection, so they move whenever the masks do
        s["fake_rate_idea"] = float(c["is_fake_idea"].mean())
        s["fake_rate_cld"] = float(c["is_fake_cld"].mean())
        s["overall_purity"] = float(c["purity"].mean())
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    # split/multi are shown because they diagnose opposite failures: split rises
    # when candidates fragment, multi rises when merging is too aggressive and
    # starts absorbing a second particle.
    hdr = (f"{'dir':<34}{'events':>8}{'cand/ev':>9}{'def1':>7}{'def2':>7}"
           f"{'split':>7}{'multi':>7}{'ghost':>7}{'ours':>7}{'clone':>7}")
    print(hdr)
    print("-" * len(hdr))
    for d in sorted(args.dirs):
        d = d.rstrip("/")
        s = backfill(d)
        if s is None:
            print(f"{os.path.basename(d):<34}  (no cache, skipped)")
            continue
        if not args.dry_run:
            with open(os.path.join(d, "summary.json"), "w") as f:
                json.dump(s, f, indent=2)
        i = s["idea"]
        g = (f"{100 * s['ghost_rate']:6.1f}%{100 * s['fake_rate_idea']:6.1f}%"
             f"{100 * s['clone_rate']:6.1f}%") if "ghost_rate" in s else f"{'truth-derived':>21}"
        print(f"{os.path.basename(d):<34}{s['n_events']:>8,}"
              f"{s.get('candidates_per_event', 0):>9.1f}"
              f"{100 * i['match_rate']:6.1f}%{100 * i['def2']:6.1f}%"
              f"{100 * i['split']:6.1f}%{100 * i['multiple']:6.1f}%{g}")


if __name__ == "__main__":
    main()
