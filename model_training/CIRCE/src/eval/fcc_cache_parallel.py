"""Parallel version of fcc_cache_at_op.py.

Two improvements over the original:
  1. Event-loop parallelised across --workers processes (embarrassingly parallel).
  2. fake-flag computation vectorised with np.isin instead of a Python for-loop.

Extra flag:
  --beta_mode sigmoid   use stored sig_beta as-is  (default, same as original)
  --beta_mode raw       apply logit transform first: beta_raw = log(p/(1-p))
                        Threshold --tbeta is then in logit-space.
                        Equivalent sigmoid threshold = sigmoid(tbeta).
                        e.g. --tbeta -3.664 with raw == --tbeta 0.025 with sigmoid.
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import polars as pl

from src.eval_fcc_metrics_v36 import (
    add_reconstructable_masks,
    per_cluster_records,
    per_track_records,
)
from src.eval_sweep_v33 import get_clustering_greedy
from src.lowpt_op_sweep import cache_from_dataframe


# ---------------------------------------------------------------------------
# Worker function (must be top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _process_chunk(args):
    """Cluster one chunk of events; return (tracks, clusters) lists."""
    chunk, tbeta, td, beta_mode = args
    tracks, clusters = [], []
    for e in chunk:
        betas = e["sig_beta"]
        if beta_mode == "raw":
            betas = np.clip(betas, 1e-7, 1.0 - 1e-7)
            betas = np.log(betas / (1.0 - betas))
        labels = get_clustering_greedy(betas, e["sig_coords"], tbeta=tbeta, td=td)
        for t in per_track_records(labels, e["sig_mc"], e["n_hits_total_map"]):
            t["event_id"] = e["event_id"]
            t["seed"] = e["seed"]
            tracks.append(t)
        for c in per_cluster_records(labels, e["sig_mc"]):
            c["event_id"] = e["event_id"]
            c["seed"] = e["seed"]
            clusters.append(c)
    return tracks, clusters


# ---------------------------------------------------------------------------
# Vectorised fake-flag helper
# ---------------------------------------------------------------------------

# Bit layout for the (mc_idx, event_id, seed) → int64 key.
# seed   occupies bits  0-12  (up to 8 191)
# event  occupies bits 13-29  (up to 131 071)
# mc_idx occupies bits 30+    (up to ~8 billion before int64 overflow)
_SEED_BITS  = 13
_EVENT_BITS = 17


def _encode_keys(mc_arr, ev_arr, sd_arr):
    return (
        mc_arr.astype(np.int64) << (_SEED_BITS + _EVENT_BITS)
        | ev_arr.astype(np.int64) << _SEED_BITS
        | sd_arr.astype(np.int64)
    )


def _reco_to_keys(reco_set):
    if not reco_set:
        return np.empty(0, dtype=np.int64)
    mc_arr = np.array([m for m, _, _ in reco_set], dtype=np.int64)
    ev_arr = np.array([e for _, e, _ in reco_set], dtype=np.int64)
    sd_arr = np.array([s for _, _, s in reco_set], dtype=np.int64)
    return _encode_keys(mc_arr, ev_arr, sd_arr)


def fake_flags_vec(purs, cl_mc, cl_ev, cl_sd, reco_set):
    """Vectorised replacement for the original fake_array loop.

    A cluster is fake when purity < 0.75  OR  its best-match particle is not
    in the reconstructable set.
    """
    low_purity = purs < 0.75
    cluster_keys = _encode_keys(cl_mc, cl_ev, cl_sd)
    reco_keys    = _reco_to_keys(reco_set)
    in_reco      = np.isin(cluster_keys, reco_keys)
    return low_purity | ~in_reco


# ---------------------------------------------------------------------------
# Summary helper (identical to original)
# ---------------------------------------------------------------------------

def overall(df: pl.DataFrame, mask_col):
    if mask_col is not None:
        df = df.filter(pl.col(mask_col))
    n = len(df)
    if n == 0:
        return {"n": 0, "efficiency": 0.0, "match_rate": 0.0}
    return {
        "n": n,
        "efficiency": float(df["efficiency_per_hit"].mean()),
        "match_rate": float(df["matched"].cast(pl.Float64).mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache_path", required=True,  help="dir with forward_hits.parquet")
    ap.add_argument("--mc_signal",  required=True,  help="compact mc_signal.parquet")
    ap.add_argument("--embed_dim",  type=int, required=True)
    ap.add_argument("--tbeta",      type=float, required=True)
    ap.add_argument("--td",         type=float, required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--tag",        default="op")
    ap.add_argument("--beta_mode",  choices=["sigmoid", "raw"], default="sigmoid",
                    help="sigmoid=use cached sig_beta as-is; raw=apply logit transform first")
    ap.add_argument("--workers",    type=int, default=32,
                    help="number of parallel worker processes")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()

    # Print threshold info so the user can judge the operating point
    if args.beta_mode == "raw":
        sig_equiv = 1.0 / (1.0 + math.exp(-args.tbeta))
        print(f"[beta_mode=raw]  tbeta={args.tbeta:.4f} in logit-space  "
              f"(equivalent sigmoid threshold = {sig_equiv:.4f})")
    else:
        raw_equiv = math.log(args.tbeta / (1.0 - args.tbeta))
        print(f"[beta_mode=sigmoid]  tbeta={args.tbeta:.4f}  "
              f"(equivalent raw/logit threshold = {raw_equiv:.4f})")

    print(f"Loading cache from {args.cache_path} ...")
    hits_df = pl.read_parquet(os.path.join(args.cache_path, "forward_hits.parquet"))
    cache = cache_from_dataframe(hits_df, args.embed_dim)
    del hits_df  # free before forking so workers don't copy-on-write it
    print(f"Loaded {len(cache)} events | workers={args.workers} | "
          f"~{len(cache) // args.workers} events/worker")

    # Split cache into chunks (one per worker)
    n_workers   = args.workers
    chunk_size  = max(1, (len(cache) + n_workers - 1) // n_workers)
    chunks      = [cache[i:i + chunk_size] for i in range(0, len(cache), chunk_size)]
    task_args   = [(chunk, args.tbeta, args.td, args.beta_mode) for chunk in chunks]
    del cache  # workers have their own copy via fork

    print(f"Clustering {len(chunks)} chunks in parallel ...")
    tracks, clusters = [], []
    with Pool(processes=n_workers) as pool:
        for i, (t, c) in enumerate(pool.imap_unordered(_process_chunk, task_args)):
            tracks.extend(t)
            clusters.extend(c)
            elapsed = time.time() - t0
            print(f"  chunk {i+1:>3}/{len(chunks)}  "
                  f"tracks={len(tracks):>9,}  clusters={len(clusters):>9,}  "
                  f"elapsed={elapsed:.0f}s",
                  flush=True)

    t_cluster = time.time() - t0
    print(f"Clustering done in {t_cluster:.0f}s  "
          f"({len(tracks):,} tracks, {len(clusters):,} clusters)")

    # ------------------------------------------------------------------
    # Join with mc_signal
    # ------------------------------------------------------------------
    mc_df = pl.read_parquet(args.mc_signal)
    print(f"Loaded {mc_df.height:,} signal mc rows")

    track_pl = pl.DataFrame(tracks)
    joined = track_pl.join(
        mc_df,
        left_on=["mc_idx", "event_id", "seed"],
        right_on=["mc_index", "event_id", "seed"],
        how="left",
    ).filter(pl.col("pt").is_not_null())
    joined = add_reconstructable_masks(joined)

    cluster_pl = pl.DataFrame(clusters)
    cluster_joined = cluster_pl.join(
        mc_df.select(["mc_index", "event_id", "seed", "pt", "theta", "phi",
                      "gen_status", "charge", "decayed_in_tracker"]),
        left_on=["matched_mc_idx", "event_id", "seed"],
        right_on=["mc_index", "event_id", "seed"],
        how="left",
    )

    # ------------------------------------------------------------------
    # Fake flags — vectorised
    # ------------------------------------------------------------------
    def reco_set(mask_col):
        f = joined.filter(pl.col(mask_col))
        return set(zip(f["mc_idx"].to_list(), f["event_id"].to_list(), f["seed"].to_list()))

    reco_idea = reco_set("is_reconstructable_idea")
    reco_cld  = reco_set("is_reconstructable_cld")

    purs  = cluster_joined["purity"].to_numpy()
    cl_mc = cluster_joined["matched_mc_idx"].to_numpy()
    cl_ev = cluster_joined["event_id"].to_numpy()
    cl_sd = cluster_joined["seed"].to_numpy()

    print(f"Computing fake flags (vectorised) over {len(purs):,} clusters ...")
    cluster_joined = cluster_joined.with_columns([
        pl.Series("is_fake_idea", fake_flags_vec(purs, cl_mc, cl_ev, cl_sd, reco_idea)),
        pl.Series("is_fake_cld",  fake_flags_vec(purs, cl_mc, cl_ev, cl_sd, reco_cld)),
    ])

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    out_tracks   = os.path.join(args.output_dir, "cache.parquet")
    out_clusters = os.path.join(args.output_dir, "cache_clusters.parquet")
    joined.write_parquet(out_tracks)
    cluster_joined.write_parquet(out_clusters)

    summary = {
        "tag":             args.tag,
        "beta_mode":       args.beta_mode,
        "tbeta":           args.tbeta,
        "td":              args.td,
        "n_tracks_total":  len(joined),
        "n_clusters_total": len(cluster_joined),
        "no_cuts":   overall(joined, None),
        "idea":      overall(joined, "is_reconstructable_idea"),
        "cld":       overall(joined, "is_reconstructable_cld"),
        "displaced": overall(joined, "is_reconstructable_displaced"),
        "fake_rate_idea": int(cluster_joined["is_fake_idea"].sum()) / max(len(cluster_joined), 1),
        "fake_rate_cld":  int(cluster_joined["is_fake_cld"].sum())  / max(len(cluster_joined), 1),
        "overall_purity": float(cluster_joined["purity"].mean()),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"FCC summary  ({args.tag}  beta_mode={args.beta_mode}  "
          f"tbeta={args.tbeta} td={args.td})")
    print("=" * 70)
    print(f"tracks={summary['n_tracks_total']:,}  clusters={summary['n_clusters_total']:,}")
    print(f"  overall purity:  {summary['overall_purity']:.3f}")
    print(f"  fake rate IDEA:  {summary['fake_rate_idea']:.3f}   "
          f"CLD: {summary['fake_rate_cld']:.3f}")
    for tag in ["no_cuts", "idea", "cld", "displaced"]:
        s = summary[tag]
        print(f"  [{tag:>9}] n={s['n']:>8,}  "
              f"eff_hit={s['efficiency']:.3f}  match_rate={s['match_rate']:.3f}")
    print(f"\nTotal wall time: {time.time() - t0:.0f}s")
    print(f"Saved {out_tracks}, {out_clusters}, summary.json")


if __name__ == "__main__":
    main()
