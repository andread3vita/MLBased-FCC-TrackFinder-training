"""Inference-time OP sweep for low-pT efficiency on a single checkpoint.

Strategy: run the forward pass exactly once per event (the expensive step),
cache the per-hit (sig_coords, sig_beta, sig_mc) outputs to disk, then
iterate the cheap greedy-clustering step over a grid of (tbeta, td)
operating points. For each OP we compute per-pT IDEA match rate and the
failure-mode mix.

The disk cache is reusable across runs:
  - First run with --cache_path X writes a single parquet with all
    forward outputs (sig_hits) and a JSON event manifest.
  - Subsequent runs with --cache_path X reuse the cache and skip the
    forward pass entirely (no GPU needed). Per-OP clustering takes
    ~10-15 minutes per OP for a 5000-event eval set.

Output:
  <out_dir>/op_summary.csv         — match rate per pT bin per OP
  <out_dir>/op_failure_modes.csv   — failure-mode mix at the lowest pT bin
  <out_dir>/eff_vs_pt_by_op.png    — overlay of eff vs pT for every OP
  <out_dir>/best_op_recommendation.md

Cache layout (when --cache_path is set):
  <cache_path>/forward_hits.parquet   one row per signal hit
                                      (event_id, seed, coord_0..D, beta,
                                       mc_index, n_hits_total)
  <cache_path>/manifest.json          {"checkpoint","embed_dim","seeds",
                                       "max_hits","n_events"}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval_sweep_v33 import (  # noqa: E402
    CGATrParquetModel,
    get_clustering_greedy,
    parse_seed_range,
)
from src.eval_fcc_metrics_v36 import (  # noqa: E402
    add_reconstructable_masks,
    load_mc_particles,
    per_track_records,
    per_cluster_records,
)
from src.dataset.parquet_dataset import IDEAParquetDataset  # noqa: E402


def cache_to_dataframe(cache, embed_dim) -> pl.DataFrame:
    """Flatten in-memory cache to a single hit-level polars DataFrame.

    Layout: one row per signal hit. Schema:
      event_id, seed, coord_0..coord_{D-1}, beta, mc_index, n_hits_total
    `n_hits_total` is the count of all (signal+secondary) hits with this
    mc_idx in the event; we denormalize so each hit knows the cut count
    and we don't need a side table.
    """
    rows = {}
    rows["event_id"] = []
    rows["seed"] = []
    for d in range(embed_dim):
        rows[f"coord_{d}"] = []
    rows["beta"] = []
    rows["mc_index"] = []
    rows["n_hits_total"] = []

    for entry in cache:
        n_sig = entry["sig_coords"].shape[0]
        rows["event_id"].extend([entry["event_id"]] * n_sig)
        rows["seed"].extend([entry["seed"]] * n_sig)
        for d in range(embed_dim):
            rows[f"coord_{d}"].extend(entry["sig_coords"][:, d].tolist())
        rows["beta"].extend(entry["sig_beta"].tolist())
        rows["mc_index"].extend(entry["sig_mc"].tolist())
        nht = entry["n_hits_total_map"]
        rows["n_hits_total"].extend([int(nht.get(int(m), 0))
                                     for m in entry["sig_mc"].tolist()])

    schema = {"event_id": pl.Int64, "seed": pl.Int64}
    for d in range(embed_dim):
        schema[f"coord_{d}"] = pl.Float32
    schema["beta"] = pl.Float32
    schema["mc_index"] = pl.Int64
    schema["n_hits_total"] = pl.Int64
    return pl.DataFrame(rows, schema=schema)


def cache_from_dataframe(df: pl.DataFrame, embed_dim: int) -> list[dict]:
    """Inverse of `cache_to_dataframe`. Splits hit-level DataFrame back into
    per-event entries with numpy arrays.
    """
    coord_cols = [f"coord_{d}" for d in range(embed_dim)]
    cache: list[dict] = []
    by_event = df.group_by(["seed", "event_id"], maintain_order=True).agg(
        [pl.col(c) for c in coord_cols]
        + [pl.col("beta"), pl.col("mc_index"), pl.col("n_hits_total")]
    )
    for row in by_event.iter_rows(named=True):
        coords = np.stack([np.asarray(row[c], dtype=np.float32) for c in coord_cols], axis=1)
        beta = np.asarray(row["beta"], dtype=np.float32)
        mc = np.asarray(row["mc_index"], dtype=np.int64)
        nht = np.asarray(row["n_hits_total"], dtype=np.int64)
        n_hits_total_map = {int(m): int(n) for m, n in zip(mc, nht)}
        cache.append({
            "event_id": int(row["event_id"]),
            "seed": int(row["seed"]),
            "sig_coords": coords,
            "sig_beta": beta,
            "sig_mc": mc,
            "n_hits_total_map": n_hits_total_map,
        })
    return cache


@torch.no_grad()
def forward_and_cache(model, dataset, device, max_events, embed_dim, log_every=200):
    """Run the model forward pass on every event once and cache the per-hit
    outputs needed for clustering. Returns a list of dict-per-event records.
    """
    model.eval()
    cache: list[dict] = []
    n_events = min(max_events, len(dataset))
    n_skipped = 0
    t0 = time.time()

    for idx in range(n_events):
        event = dataset[idx]
        if event is None:
            n_skipped += 1
            continue

        features = event["features"].to(device)
        mc_index_all = event["mc_index"].numpy()
        is_secondary = event["is_secondary"].numpy().astype(bool)
        seq_lens = [event["n_hits"]]

        sig_mask = (~is_secondary) & (mc_index_all != 0)
        if sig_mask.sum() < 4:
            n_skipped += 1
            continue

        try:
            output = model(features, seq_lens)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            n_skipped += 1
            continue

        coords = output[:, :embed_dim].cpu().numpy().astype(np.float32)
        beta = torch.sigmoid(output[:, embed_dim]).cpu().numpy().astype(np.float32)

        sig_coords = coords[sig_mask]
        sig_beta = beta[sig_mask]
        sig_mc = mc_index_all[sig_mask]

        n_hits_total_map = {}
        unique_mc_all = np.unique(mc_index_all)
        unique_mc_all = unique_mc_all[unique_mc_all > 0]
        for mc_idx in unique_mc_all:
            n_hits_total_map[int(mc_idx)] = int((mc_index_all == mc_idx).sum())

        dc_path, _vtx, eid, _ = dataset._index[idx]
        seed = int(Path(dc_path).parent.name.replace("seed_", ""))

        cache.append({
            "event_id": int(eid),
            "seed": int(seed),
            "sig_coords": sig_coords,
            "sig_beta": sig_beta,
            "sig_mc": sig_mc,
            "n_hits_total_map": n_hits_total_map,
        })

        del output, coords, beta
        if (idx + 1) % 200 == 0:
            torch.cuda.empty_cache()

        if (idx + 1) % log_every == 0 or idx + 1 == n_events:
            dt = time.time() - t0
            rate = (idx + 1) / max(dt, 1e-3)
            eta = (n_events - idx - 1) / max(rate, 1e-3)
            print(
                f"  Forward {idx + 1}/{n_events}  "
                f"{rate:.2f} ev/s  ETA {eta / 60.0:.1f} min  skipped={n_skipped}",
                flush=True,
            )

    print(f"Forward done in {(time.time() - t0) / 60.0:.1f} min, "
          f"cached {len(cache)} events ({n_skipped} skipped)")
    return cache


def cluster_and_records(cache, tbeta, td):
    """Run greedy clustering for a single (tbeta, td) OP across the cached
    forward outputs. Returns a polars dataframe with track records.
    """
    track_records: list[dict] = []
    cluster_records: list[dict] = []

    for entry in cache:
        labels = get_clustering_greedy(
            entry["sig_beta"], entry["sig_coords"], tbeta=tbeta, td=td,
        )
        track_rows = per_track_records(labels, entry["sig_mc"], entry["n_hits_total_map"])
        cluster_rows = per_cluster_records(labels, entry["sig_mc"])
        for tr in track_rows:
            tr["event_id"] = entry["event_id"]
            tr["seed"] = entry["seed"]
            track_records.append(tr)
        for cr in cluster_rows:
            cr["event_id"] = entry["event_id"]
            cr["seed"] = entry["seed"]
            cluster_records.append(cr)

    return track_records, cluster_records


def per_pt_match_rate(joined: pl.DataFrame, pt_bins: list[float],
                      eff_thresh: float = 0.50) -> list[dict]:
    """Compute IDEA match rate per pT bin under both metrics.

    The standard `matched` flag in `per_track_records` is purity-only
    (`purity_of_match >= 0.75`). That metric becomes degenerate as `td`
    shrinks (every signal hit becomes a singleton cluster of purity 1.0).

    To detect that pathology we additionally compute a **strict** match
    rate that requires the matched cluster to capture at least
    `eff_thresh` of the truth track's hits in the IDEA cut.
    """
    idea = joined.filter(pl.col("is_reconstructable_idea"))
    idea = idea.with_columns(
        (pl.col("matched") & (pl.col("efficiency_per_hit") >= eff_thresh))
        .alias("matched_strict")
    )
    out = []
    for lo, hi in zip(pt_bins[:-1], pt_bins[1:]):
        sub = idea.filter((pl.col("pt") >= lo) & (pl.col("pt") < hi))
        n = sub.height
        m = sub.filter(pl.col("matched")).height
        m_strict = sub.filter(pl.col("matched_strict")).height
        median_eff = (
            float(sub.filter(pl.col("matched"))["efficiency_per_hit"].median())
            if m > 0 else float("nan")
        )
        out.append({
            "pt_lo": lo, "pt_hi": hi, "n": int(n),
            "matched": int(m),
            "matched_strict": int(m_strict),
            "match_rate": (m / n) if n > 0 else float("nan"),
            "match_rate_strict": (m_strict / n) if n > 0 else float("nan"),
            "median_eff_of_matched": median_eff,
        })
    return out


def failure_mode_mix(joined: pl.DataFrame, cluster_joined: pl.DataFrame,
                     pt_lo: float, pt_hi: float) -> dict:
    """Failure-mode mix at a single pT bin (uses the same definitions as
    `lowpt_failure_diagnostic.py`)."""
    frag_count = (
        cluster_joined.filter(pl.col("matched_mc_idx") > 0)
        .group_by(["seed", "event_id", "matched_mc_idx"])
        .agg(pl.len().alias("n_dominant_clusters"))
        .rename({"matched_mc_idx": "mc_idx"})
    )
    annotated = joined.join(
        frag_count, on=["seed", "event_id", "mc_idx"], how="left"
    ).with_columns(pl.col("n_dominant_clusters").fill_null(0))

    sub = annotated.filter(
        pl.col("is_reconstructable_idea")
        & (pl.col("pt") >= pt_lo)
        & (pl.col("pt") < pt_hi)
    )
    n_total = sub.height
    if n_total == 0:
        return {"n_total": 0}

    failed = sub.filter(~pl.col("matched"))
    n_failed = failed.height
    if n_failed == 0:
        return {"n_total": n_total, "n_failed": 0,
                "match_rate": 1.0,
                "by_failure_mode_pct": {k: 0.0 for k in
                                        ["SUPPRESSION", "FRAGMENTATION",
                                         "THRESHOLD", "CONFUSION"]}}

    suppr = failed.filter(
        (pl.col("best_label") == -1) | (pl.col("efficiency_per_hit") < 0.20)
    ).height
    frag = failed.filter(
        (pl.col("efficiency_per_hit") < 0.75) & (pl.col("n_dominant_clusters") >= 2)
    ).height
    thresh = failed.filter(
        (pl.col("purity_of_match") >= 0.65) & (pl.col("purity_of_match") < 0.75)
        & (pl.col("efficiency_per_hit") >= 0.5)
    ).height
    suppr_set = failed.filter(
        (pl.col("best_label") == -1) | (pl.col("efficiency_per_hit") < 0.20)
    )["mc_idx"].to_list()
    frag_only = failed.filter(
        ~((pl.col("best_label") == -1) | (pl.col("efficiency_per_hit") < 0.20))
        & (pl.col("efficiency_per_hit") < 0.75)
        & (pl.col("n_dominant_clusters") >= 2)
    ).height
    thresh_only = failed.filter(
        ~((pl.col("best_label") == -1) | (pl.col("efficiency_per_hit") < 0.20))
        & ~((pl.col("efficiency_per_hit") < 0.75) & (pl.col("n_dominant_clusters") >= 2))
        & (pl.col("purity_of_match") >= 0.65) & (pl.col("purity_of_match") < 0.75)
        & (pl.col("efficiency_per_hit") >= 0.5)
    ).height
    conf = n_failed - suppr - frag_only - thresh_only

    return {
        "n_total": n_total,
        "n_failed": n_failed,
        "match_rate": 1.0 - n_failed / n_total,
        "by_failure_mode_pct": {
            "SUPPRESSION": suppr / n_failed,
            "FRAGMENTATION": frag_only / n_failed,
            "THRESHOLD": thresh_only / n_failed,
            "CONFUSION": conf / n_failed,
        },
    }


def join_with_mc(track_records, cluster_records, mc_df) -> tuple[pl.DataFrame, pl.DataFrame]:
    track_pl = pl.DataFrame(track_records)
    joined = track_pl.join(
        mc_df,
        left_on=["mc_idx", "event_id", "seed"],
        right_on=["mc_index", "event_id", "seed"],
        how="left",
    ).filter(pl.col("pt").is_not_null())
    joined = add_reconstructable_masks(joined)

    cluster_pl = pl.DataFrame(cluster_records)
    cluster_joined = cluster_pl.join(
        mc_df.select([
            "mc_index", "event_id", "seed", "pt", "theta", "phi",
            "gen_status", "charge", "decayed_in_tracker",
        ]),
        left_on=["matched_mc_idx", "event_id", "seed"],
        right_on=["mc_index", "event_id", "seed"],
        how="left",
    )
    return joined, cluster_joined


def fake_rate_idea(joined, cluster_joined) -> float:
    """Cluster-level IDEA fake rate for a single OP."""
    reco_idea_set = set(zip(
        joined.filter(pl.col("is_reconstructable_idea"))["mc_idx"].to_list(),
        joined.filter(pl.col("is_reconstructable_idea"))["event_id"].to_list(),
        joined.filter(pl.col("is_reconstructable_idea"))["seed"].to_list(),
    ))
    purs = cluster_joined["purity"].to_numpy()
    cl_mc = cluster_joined["matched_mc_idx"].to_numpy()
    cl_ev = cluster_joined["event_id"].to_numpy()
    cl_sd = cluster_joined["seed"].to_numpy()
    n_total = len(purs)
    if n_total == 0:
        return float("nan")
    n_fake = 0
    for i in range(n_total):
        if purs[i] < 0.75:
            n_fake += 1
        elif (int(cl_mc[i]), int(cl_ev[i]), int(cl_sd[i])) not in reco_idea_set:
            n_fake += 1
    return n_fake / n_total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval_seeds", default="181-190")
    p.add_argument("--max_hits", type=int, default=20000)
    p.add_argument("--max_events", type=int, default=5000)
    p.add_argument("--embed_dim", type=int, default=5)
    p.add_argument("--num_blocks", type=int, default=10)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tag", default="v36ef")
    p.add_argument("--tbeta_grid", nargs="+", type=float,
                   default=[0.025, 0.05])
    p.add_argument("--td_grid", nargs="+", type=float,
                   default=[0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.30])
    p.add_argument(
        "--cache_path", type=str, default=None,
        help="Directory for the on-disk forward-pass cache. If the cache "
             "exists and is compatible, the GPU forward pass is skipped. If "
             "the cache does not exist, it is created after the forward pass. "
             "Cache contents: forward_hits.parquet + manifest.json.",
    )
    p.add_argument(
        "--force_refresh_cache", action="store_true",
        help="If set with --cache_path, ignore any existing cache and "
             "overwrite it with a fresh forward pass.",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cache: list[dict] | None = None
    cache_used = False
    if args.cache_path is not None and not args.force_refresh_cache:
        hits_path = os.path.join(args.cache_path, "forward_hits.parquet")
        manifest_path = os.path.join(args.cache_path, "manifest.json")
        if os.path.exists(hits_path) and os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            same_ckpt = manifest.get("checkpoint") == args.checkpoint
            same_seeds = manifest.get("seeds") == args.eval_seeds
            same_dim = manifest.get("embed_dim") == args.embed_dim
            same_max_hits = manifest.get("max_hits") == args.max_hits
            same_max_events = manifest.get("max_events") == args.max_events
            if same_ckpt and same_seeds and same_dim and same_max_hits and same_max_events:
                print(f"\n=== Reusing forward cache from {args.cache_path} ===")
                t0 = time.time()
                hits_df = pl.read_parquet(hits_path)
                cache = cache_from_dataframe(hits_df, args.embed_dim)
                print(
                    f"  loaded {len(cache)} events ({hits_df.height} hits) "
                    f"in {time.time() - t0:.1f}s"
                )
                cache_used = True
            else:
                print(
                    "\n=== Cache exists but manifest mismatch; regenerating "
                    f"(ckpt={same_ckpt} seeds={same_seeds} dim={same_dim} "
                    f"max_hits={same_max_hits} max_events={same_max_events}) ==="
                )

    if not cache_used:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        print(f"Loading checkpoint: {args.checkpoint}")
        model = CGATrParquetModel(num_blocks=args.num_blocks, embed_dim=args.embed_dim)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            if "ema_state_dict" in state and state["ema_state_dict"] is not None:
                state = state["ema_state_dict"]
            else:
                sd = state["state_dict"]
                state = {k[len("model."):] if k.startswith("model.") else k: v
                         for k, v in sd.items()}
        model.load_state_dict(state, strict=True)
        model = model.to(device)
        print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

        seed_start, seed_end = parse_seed_range(args.eval_seeds)
        print(f"Loading eval data: seeds {seed_start}-{seed_end - 1}")
        dataset = IDEAParquetDataset(
            args.data_dir,
            seed_range=(seed_start, seed_end),
            max_hits_per_event=args.max_hits,
        )
        print(f"Dataset: {len(dataset)} events (max_hits={args.max_hits})")

        print("\n=== Phase 1: forward pass + cache ===")
        cache = forward_and_cache(
            model, dataset, device, args.max_events, args.embed_dim,
        )

        del model
        torch.cuda.empty_cache()

        if args.cache_path is not None:
            os.makedirs(args.cache_path, exist_ok=True)
            print(f"\nSaving forward cache to {args.cache_path}...")
            t0 = time.time()
            hits_df = cache_to_dataframe(cache, args.embed_dim)
            hits_df.write_parquet(
                os.path.join(args.cache_path, "forward_hits.parquet"),
                compression="zstd",
            )
            with open(os.path.join(args.cache_path, "manifest.json"), "w") as f:
                json.dump({
                    "checkpoint": args.checkpoint,
                    "seeds": args.eval_seeds,
                    "embed_dim": args.embed_dim,
                    "max_hits": args.max_hits,
                    "max_events": args.max_events,
                    "n_events": len(cache),
                    "n_hits": int(hits_df.height),
                }, f, indent=2)
            print(
                f"  cache written ({hits_df.height} hits, "
                f"{os.path.getsize(os.path.join(args.cache_path, 'forward_hits.parquet')) / 1e6:.1f} MB) "
                f"in {time.time() - t0:.1f}s"
            )

    seed_start, seed_end = parse_seed_range(args.eval_seeds)

    print(f"\nLoading mc_particles for seeds {seed_start}-{seed_end - 1}...")
    mc_df = load_mc_particles(args.data_dir, seed_start, seed_end)
    print(f"  loaded {len(mc_df)} particle rows")

    pt_bins = [0.10, 0.15, 0.20, 0.30, 0.50, 1.00, 3.00, 30.0]
    op_grid = [(tb, td) for tb in args.tbeta_grid for td in args.td_grid]

    summary_rows: list[dict] = []
    failure_rows: list[dict] = []
    eff_curves: dict[str, list[dict]] = {}

    print(f"\n=== Phase 2: cluster + score over {len(op_grid)} OPs ===")
    for i, (tb, td) in enumerate(op_grid):
        t0 = time.time()
        track_records, cluster_records = cluster_and_records(cache, tb, td)
        joined, cluster_joined = join_with_mc(track_records, cluster_records, mc_df)
        n_clusters = len(cluster_records)
        per_pt = per_pt_match_rate(joined, pt_bins)
        fr = failure_mode_mix(joined, cluster_joined, 0.10, 0.15)
        fake = fake_rate_idea(joined, cluster_joined)

        op_label = f"tb{tb}_td{td}"
        eff_curves[op_label] = per_pt
        for row in per_pt:
            summary_rows.append({
                "tbeta": tb, "td": td, "op": op_label,
                "pt_lo": row["pt_lo"], "pt_hi": row["pt_hi"],
                "n": row["n"], "matched": row["matched"],
                "matched_strict": row["matched_strict"],
                "match_rate": row["match_rate"],
                "match_rate_strict": row["match_rate_strict"],
                "median_eff_of_matched": row["median_eff_of_matched"],
                "n_clusters": n_clusters,
                "fake_rate_idea": fake,
            })

        idea = joined.filter(pl.col("is_reconstructable_idea"))
        idea_strict = idea.with_columns(
            (pl.col("matched") & (pl.col("efficiency_per_hit") >= 0.50))
            .alias("matched_strict")
        )
        n_idea = idea.height
        idea_match_overall = idea.filter(pl.col("matched")).height / max(n_idea, 1)
        idea_match_strict_overall = (
            idea_strict.filter(pl.col("matched_strict")).height / max(n_idea, 1)
        )
        median_eff_overall = (
            float(idea.filter(pl.col("matched"))["efficiency_per_hit"].median())
            if idea.filter(pl.col("matched")).height > 0 else float("nan")
        )
        failure_rows.append({
            "tbeta": tb, "td": td, "op": op_label,
            "match_010_015": fr.get("match_rate", float("nan")),
            "match_idea_total": idea_match_overall,
            "match_idea_strict_total": idea_match_strict_overall,
            "median_eff_of_matched": median_eff_overall,
            "fake_rate_idea": fake,
            "n_clusters": n_clusters,
            **{f"f_{k}": v for k, v in fr.get("by_failure_mode_pct", {}).items()},
        })

        dt = time.time() - t0
        m_lowpt = per_pt[0]["match_rate"]
        m_lowpt_strict = per_pt[0]["match_rate_strict"]
        m_high = per_pt[-1]["match_rate"]
        print(
            f"  [{i + 1}/{len(op_grid)}] tbeta={tb} td={td}  "
            f"low-pT match={m_lowpt:.3f} (strict={m_lowpt_strict:.3f})  "
            f"high-pT={m_high:.3f}  median_eff={median_eff_overall:.3f}  "
            f"IDEA fake={fake*100:.2f}%  n_cl={n_clusters/1e6:.2f}M  ({dt:.1f}s)"
        )

    summary_df = pl.DataFrame(summary_rows)
    failure_df = pl.DataFrame(failure_rows)
    summary_df.write_csv(os.path.join(args.out_dir, "op_summary.csv"))
    failure_df.write_csv(os.path.join(args.out_dir, "op_failure_modes.csv"))
    print(f"\nSaved op_summary.csv ({summary_df.height} rows)")
    print(f"Saved op_failure_modes.csv ({failure_df.height} rows)")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    pt_centers = [0.5 * (b["pt_lo"] + b["pt_hi"]) for b in eff_curves[next(iter(eff_curves))]]
    cmap = plt.colormaps["viridis"]
    n_curves = len(eff_curves)
    for i, (op_label, rows) in enumerate(eff_curves.items()):
        match_loose = [r["match_rate"] for r in rows]
        match_strict = [r["match_rate_strict"] for r in rows]
        col = cmap(i / max(n_curves - 1, 1))
        axes[0].plot(pt_centers, match_loose, marker="o", color=col, linewidth=1.0,
                     label=op_label)
        axes[1].plot(pt_centers, match_strict, marker="o", color=col, linewidth=1.0,
                     label=op_label)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("pT [GeV] (bin centre)")
        ax.grid(alpha=0.3)
        ax.set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Tracking efficiency, IDEA cuts")
    axes[0].set_title(f"{args.tag}: loose match (purity \u2265 0.75)")
    axes[1].set_title(f"{args.tag}: strict match (purity \u2265 0.75 AND eff \u2265 0.50)")
    axes[1].legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    plot_path = os.path.join(args.out_dir, "eff_vs_pt_by_op.png")
    fig.savefig(plot_path, dpi=130)
    print(f"Saved {plot_path}")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    tds = sorted({r["td"] for r in failure_rows})
    by_td = {r["td"]: r for r in failure_rows}
    loose = [by_td[td]["match_idea_total"] for td in tds]
    strict = [by_td[td]["match_idea_strict_total"] for td in tds]
    med_eff = [by_td[td]["median_eff_of_matched"] for td in tds]
    fake = [by_td[td]["fake_rate_idea"] for td in tds]
    ax2.plot(tds, loose, marker="o", label="match (loose, purity\u22650.75)")
    ax2.plot(tds, strict, marker="s", label="match (strict, +eff\u22650.50)")
    ax2.plot(tds, med_eff, marker="^", linestyle="--", label="median eff_per_hit of matched")
    ax2.plot(tds, fake, marker="x", linestyle=":", label="IDEA fake rate")
    ax2.set_xlabel("td (clustering threshold)")
    ax2.set_ylabel("rate")
    ax2.set_title(f"{args.tag}: metrics vs td (loose vs strict matching)")
    ax2.set_ylim(0.0, 1.02)
    ax2.set_xscale("log")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="best")
    fig2.tight_layout()
    plot2_path = os.path.join(args.out_dir, "metrics_vs_td.png")
    fig2.savefig(plot2_path, dpi=130)
    print(f"Saved {plot2_path}")

    print("\n=== Best-OP recommendation ===")
    failure_df_sorted = failure_df.sort("match_010_015", descending=True)
    print(failure_df_sorted)
    rec_path = os.path.join(args.out_dir, "best_op_recommendation.md")
    with open(rec_path, "w") as f:
        f.write(f"# OP sweep on {args.tag}\n\n")
        f.write(
            "**Strict match** = purity >= 0.75 AND efficiency_per_hit >= 0.50.\n"
            "Loose match (the FCC slide-24 metric) is purity-only and degenerates\n"
            "as td -> 0 (singleton clusters trivially pass the purity cut).\n\n"
        )
        f.write("Sorted by strict match rate (overall IDEA).\n\n")
        f.write(
            "| op | loose 0.1-0.15 | loose IDEA | strict IDEA | "
            "median eff (matched) | IDEA fake | n_cl | f_SUPPR | f_FRAG | f_THRESH | f_CONF |\n"
        )
        f.write(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        failure_df_sorted = failure_df.sort("match_idea_strict_total", descending=True)
        for r in failure_df_sorted.iter_rows(named=True):
            f.write(
                f"| {r['op']} | {r['match_010_015']*100:.2f}% | "
                f"{r['match_idea_total']*100:.2f}% | "
                f"{r['match_idea_strict_total']*100:.2f}% | "
                f"{r['median_eff_of_matched']:.3f} | "
                f"{r['fake_rate_idea']*100:.2f}% | "
                f"{r['n_clusters']/1e6:.2f}M | "
                f"{r['f_SUPPRESSION']*100:.1f}% | {r['f_FRAGMENTATION']*100:.1f}% | "
                f"{r['f_THRESHOLD']*100:.1f}% | {r['f_CONFUSION']*100:.1f}% |\n"
            )
    print(f"Saved {rec_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
