"""Re-apply the reconstructable masks to already-cached eval runs.

The IDEA reconstructable selection in `add_reconstructable_masks` was missing
the pt > 0.1 GeV requirement that the paper states, so every cached run carries
a denominator ~8% larger than the documented one (and those sub-100 MeV
particles are matched only ~67% of the time, which drags the reported match
rate down by about two points). The clustering is unaffected, so the fix does
not need a re-run: the masks are recomputed from the particle columns already
in cache.parquet, `is_fake_idea` is recomputed from the new reconstructable set,
and summary.json is rewritten.

Both parquets are rewritten in place. Nothing here depends on the model, so a
run can always be recovered by re-running fcc_cache_parallel.py.

  python src/eval/migrate_pt_cut.py --dry_run eval_results/*/merge_sweep/*/
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import polars as pl

from src.eval.backfill_summaries import backfill
from src.eval.fcc_cache_parallel import fake_flags_vec
from src.eval_fcc_metrics_v36 import add_reconstructable_masks

MASKS = ["is_reconstructable_idea", "is_reconstructable_cld",
         "is_reconstructable_displaced"]
DERIVED = ["vertex_r", "theta_deg", "eta", "cos_theta"]


def migrate(d, dry_run):
    tp = os.path.join(d, "cache.parquet")
    cp = os.path.join(d, "cache_clusters.parquet")
    if not os.path.exists(tp):
        return None
    t = pl.read_parquet(tp)
    before = {m: float(t[m].mean()) for m in MASKS if m in t.columns}
    before_match = float(t.filter(pl.col("is_reconstructable_idea"))["matched"].mean())

    t = add_reconstructable_masks(t.drop([c for c in MASKS + DERIVED
                                          if c in t.columns]))
    after = {m: float(t[m].mean()) for m in MASKS}
    r = t.filter(pl.col("is_reconstructable_idea"))
    after_match = float(r["matched"].mean())
    out = {"dir": d, "n_idea_before": int(before["is_reconstructable_idea"] * len(t)),
           "n_idea_after": len(r), "match_before": before_match,
           "match_after": after_match}

    if not dry_run:
        t.write_parquet(tp)
        if os.path.exists(cp):
            c = pl.read_parquet(cp)
            reco = set(zip(r["mc_idx"].to_list(), r["event_id"].to_list(),
                           r["seed"].to_list()))
            c = c.with_columns(pl.Series("is_fake_idea", fake_flags_vec(
                c["purity"].to_numpy(), c["matched_mc_idx"].to_numpy(),
                c["event_id"].to_numpy(), c["seed"].to_numpy(), reco)))
            c.write_parquet(cp)
        s = backfill(d)
        if s is not None:
            with open(os.path.join(d, "summary.json"), "w") as f:
                json.dump(s, f, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    hdr = f"{'dir':<46}{'n IDEA before':>15}{'after':>10}{'match before':>14}{'after':>9}"
    print(hdr)
    print("-" * len(hdr))
    for d in sorted(args.dirs):
        d = d.rstrip("/")
        o = migrate(d, args.dry_run)
        name = os.path.relpath(d).replace("eval_results/", "")
        if o is None:
            print(f"{name:<46}  (no cache, skipped)")
            continue
        print(f"{name:<46}{o['n_idea_before']:>15,}{o['n_idea_after']:>10,}"
              f"{100 * o['match_before']:13.2f}%{100 * o['match_after']:8.2f}%")


if __name__ == "__main__":
    main()
