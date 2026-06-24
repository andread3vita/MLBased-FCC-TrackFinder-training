"""Oracle-merge diagnostic: simulate a perfect cluster-merging step.

WARNING — this is **not** a tracking-efficiency metric.  It uses
ground-truth `mc_idx` labels at inference time to merge predicted
clusters that share a dominant truth particle (purity >= T_assoc).
The resulting numbers are an *upper bound on what the embedding could
deliver if its inference clustering were replaced by a perfect
merger* — the gap to the unmerged metric measures how much
inference-time algorithm work is left on the table.

Always report the unmerged number as the deployment metric and the
merged number as a clearly-labelled diagnostic.

Input: an FCC-style cache directory containing
  - cache.parquet           per-truth-track records
  - cache_clusters.parquet  per-predicted-cluster records

Output: a new directory in the same schema, with each truth track's
merged super-cluster replacing the "best cluster" stats:
  - cache.parquet           per-track records, recomputed against the
                            merged super-cluster
  - cache_clusters.parquet  super-clusters (one per associated truth
                            track) + unassociated leftover originals
  - manifest.json           merging configuration and bookkeeping

Usage:
    python -m src.truth_merge_diagnostic \\
        --in_dir  eval_results/v35_fcc_unlimited \\
        --out_dir eval_results/v35_fcc_unlimited_merge_T0.75 \\
        --threshold 0.75
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def truth_merge(
    per_track: pl.DataFrame,
    per_cluster: pl.DataFrame,
    threshold: float,
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Apply oracle-merge.

    Rules:
      - A predicted cluster c is "associated" to its `matched_mc_idx`
        iff `c.purity >= threshold` AND `c.matched_mc_idx > 0`
        (mc_idx == 0 is reserved for "no signal hits in cluster").
      - All clusters associated to the same (mc_idx, event_id, seed)
        merge into one super-cluster S_m.
      - For each truth track m, the new "best cluster" stats are the
        sums over its associated clusters:
            merged_cluster_size = sum(c.cluster_size)
            merged_overlap      = sum(c.best_match)
        which gives:
            merged_efficiency = merged_overlap / m.n_hits_signal
            merged_purity     = merged_overlap / merged_cluster_size
        (purity >= threshold by construction; 0 if no associated cluster).
      - Tracks with no associated cluster are unmatched.
      - Unassociated original clusters (purity < threshold or noise) stay
        as-is in the per-cluster table — they count toward fake rate.
    """
    associated = per_cluster.filter(
        (pl.col("purity") >= threshold) & (pl.col("matched_mc_idx") > 0)
    )

    super_aggregates = (
        associated.group_by(["matched_mc_idx", "event_id", "seed"])
        .agg(
            pl.col("cluster_size").sum().alias("merged_cluster_size"),
            pl.col("best_match").sum().alias("merged_overlap"),
            pl.col("cluster_id").count().alias("n_clusters_in_super"),
            pl.col("cluster_id").min().alias("super_label"),
        )
        .rename({"matched_mc_idx": "mc_idx"})
    )

    merged_track = per_track.join(
        super_aggregates,
        on=["mc_idx", "event_id", "seed"],
        how="left",
    ).with_columns(
        pl.col("merged_cluster_size").fill_null(0),
        pl.col("merged_overlap").fill_null(0),
        pl.col("n_clusters_in_super").fill_null(0),
        pl.col("super_label").fill_null(-1),
    )

    merged_track = merged_track.with_columns(
        (pl.col("merged_overlap") /
         pl.when(pl.col("n_hits_signal") > 0)
           .then(pl.col("n_hits_signal"))
           .otherwise(1)).alias("efficiency_per_hit_new"),
        (pl.col("merged_overlap") /
         pl.when(pl.col("merged_cluster_size") > 0)
           .then(pl.col("merged_cluster_size"))
           .otherwise(1)).alias("purity_of_match_new"),
    )

    merged_track = merged_track.with_columns(
        (
            (pl.col("purity_of_match_new") >= 0.75)
            & (pl.col("merged_overlap") > 0)
        ).alias("matched_new"),
    )

    keep_cols = [
        "mc_idx", "n_hits_signal", "n_hits_total",
        "event_id", "seed",
        "pt", "theta", "phi", "vx", "vy", "vz",
        "gen_status", "decayed_in_tracker", "charge", "pdg",
        "vertex_r", "theta_deg", "eta", "cos_theta",
        "is_reconstructable_idea", "is_reconstructable_cld",
        "is_reconstructable_displaced",
        "n_clusters_in_super",
    ]
    merged_track_out = merged_track.with_columns(
        pl.col("super_label").alias("best_label"),
        pl.col("merged_overlap").alias("best_match"),
        pl.col("merged_cluster_size").alias("cluster_size"),
        pl.col("efficiency_per_hit_new").alias("efficiency_per_hit"),
        pl.col("purity_of_match_new").alias("purity_of_match"),
        pl.col("matched_new").alias("matched"),
    ).select(
        keep_cols
        + ["best_label", "best_match", "cluster_size",
           "efficiency_per_hit", "purity_of_match", "matched"]
    )

    super_as_clusters = super_aggregates.select(
        pl.col("super_label").alias("cluster_id"),
        pl.col("merged_cluster_size").alias("cluster_size"),
        pl.col("mc_idx").alias("matched_mc_idx"),
        pl.col("merged_overlap").alias("best_match"),
        (pl.col("merged_overlap") /
         pl.when(pl.col("merged_cluster_size") > 0)
           .then(pl.col("merged_cluster_size"))
           .otherwise(1)).alias("purity"),
        pl.col("event_id"),
        pl.col("seed"),
    )

    associated_keys = associated.select(
        "cluster_id", "event_id", "seed"
    ).unique()
    leftover = per_cluster.join(
        associated_keys,
        on=["cluster_id", "event_id", "seed"],
        how="anti",
    ).select(
        "cluster_id", "cluster_size", "matched_mc_idx", "best_match",
        "purity", "event_id", "seed",
    )

    merged_clusters_no_extras = pl.concat(
        [super_as_clusters, leftover], how="vertical_relaxed",
    )

    extra_cluster_cols = [
        c for c in per_cluster.columns
        if c not in {"cluster_id", "cluster_size", "matched_mc_idx",
                     "best_match", "purity", "event_id", "seed",
                     "is_fake_idea", "is_fake_cld"}
    ]
    if extra_cluster_cols:
        cluster_extras = per_cluster.select(
            ["cluster_id", "event_id", "seed"] + extra_cluster_cols
        ).unique(subset=["cluster_id", "event_id", "seed"], keep="first")
        merged_clusters_no_extras = merged_clusters_no_extras.join(
            cluster_extras,
            on=["cluster_id", "event_id", "seed"],
            how="left",
        )

    reco_idea = (
        per_track.filter(pl.col("is_reconstructable_idea"))
        .select(
            pl.col("mc_idx").alias("matched_mc_idx"),
            pl.col("event_id"),
            pl.col("seed"),
        )
        .with_columns(pl.lit(True).alias("_reco_idea"))
    )
    reco_cld = (
        per_track.filter(pl.col("is_reconstructable_cld"))
        .select(
            pl.col("mc_idx").alias("matched_mc_idx"),
            pl.col("event_id"),
            pl.col("seed"),
        )
        .with_columns(pl.lit(True).alias("_reco_cld"))
    )

    merged_clusters = (
        merged_clusters_no_extras
        .join(reco_idea, on=["matched_mc_idx", "event_id", "seed"], how="left")
        .join(reco_cld,  on=["matched_mc_idx", "event_id", "seed"], how="left")
        .with_columns(
            pl.col("_reco_idea").fill_null(False),
            pl.col("_reco_cld").fill_null(False),
        )
        .with_columns(
            ((pl.col("purity") < 0.75) | ~pl.col("_reco_idea")).alias("is_fake_idea"),
            ((pl.col("purity") < 0.75) | ~pl.col("_reco_cld")).alias("is_fake_cld"),
        )
        .drop("_reco_idea", "_reco_cld")
    )

    n_super = super_aggregates.height
    n_leftover = leftover.height
    n_clusters_orig = per_cluster.height
    n_associated = associated.height

    bookkeeping = {
        "threshold": threshold,
        "n_clusters_original": int(n_clusters_orig),
        "n_clusters_associated": int(n_associated),
        "n_super_clusters": int(n_super),
        "n_clusters_leftover": int(n_leftover),
        "n_clusters_after_merge": int(n_super + n_leftover),
        "compression_ratio": (
            float(n_clusters_orig) / max(n_super + n_leftover, 1)
        ),
        "n_tracks": int(merged_track_out.height),
        "n_tracks_with_super": int(super_aggregates.height),
    }

    return merged_track_out, merged_clusters, bookkeeping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True,
                    help="Directory with cache.parquet + cache_clusters.parquet")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--threshold", type=float, required=True,
                    help="Cluster->mc_idx association threshold T_assoc (e.g. 0.75)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading caches from {in_dir}")
    per_track = pl.read_parquet(in_dir / "cache.parquet")
    per_cluster = pl.read_parquet(in_dir / "cache_clusters.parquet")
    print(f"  per-track:   {per_track.height} rows")
    print(f"  per-cluster: {per_cluster.height} rows")

    merged_track, merged_clusters, info = truth_merge(
        per_track, per_cluster, args.threshold,
    )
    print(f"After merge at T={args.threshold}:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    merged_track.write_parquet(out_dir / "cache.parquet", compression="zstd")
    merged_clusters.write_parquet(
        out_dir / "cache_clusters.parquet", compression="zstd",
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {"source_cache": str(in_dir), **info, "kind": "oracle_merge_diagnostic"},
            indent=2,
        )
    )
    print(f"Wrote {out_dir / 'cache.parquet'}")
    print(f"Wrote {out_dir / 'cache_clusters.parquet'}")
    print(f"Wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
