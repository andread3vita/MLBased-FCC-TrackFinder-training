"""Parameter sweep for the user-provided self-seed greedy clusterer.

Loads a saved forward cache (v35 by default), runs `get_clustering_user`
at a grid of (tbeta, td), computes FCC-style metrics per OP, and writes:
  <out_dir>/sweep.csv     overall per-OP metrics
  <out_dir>/sweep.md      markdown table
  <out_dir>/heatmaps.png  loose / strict50 / strict90 / strict99 heatmaps
  <out_dir>/per_pt_best.png  per-pT curves at best OPs vs reference

Side products (one FCC cache per "best" OP, ready for downstream
comparison via final_consolidated_comparison.py):
  <fcc_dir>/v35_fcc_user_tbeta{TB}_td{TD}/cache.parquet
  <fcc_dir>/v35_fcc_user_tbeta{TB}_td{TD}/cache_clusters.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.user_greedy import get_clustering_user
from src.lowpt_op_sweep import (
    cache_from_dataframe, join_with_mc, per_pt_match_rate, fake_rate_idea,
)
from src.eval_fcc_metrics_v36 import (
    per_track_records, per_cluster_records, load_mc_particles,
)


def overall_summary(joined: pl.DataFrame, thresholds=(0.50, 0.75, 0.90, 0.99)) -> dict:
    idea = joined.filter(pl.col("is_reconstructable_idea"))
    n_idea = max(idea.height, 1)
    n_loose = idea.filter(pl.col("matched")).height
    matched = idea.filter(pl.col("matched"))
    out = {
        "n_idea": idea.height,
        "loose_idea": n_loose / n_idea,
    }
    for T in thresholds:
        n = matched.filter(pl.col("efficiency_per_hit") >= T).height
        out[f"strict{int(T*100)}_idea"] = n / n_idea
    out["median_eff_idea"] = (
        float(matched["efficiency_per_hit"].median()) if matched.height > 0
        else float("nan")
    )
    return out


def run_op(cache, tbeta, td, log_every=2000):
    track_records = []
    cluster_records = []
    n_evt = len(cache)
    t0 = time.time()
    for i, entry in enumerate(cache):
        labels = get_clustering_user(
            entry["sig_beta"], entry["sig_coords"], tbeta=tbeta, td=td,
        )
        for r in per_track_records(labels, entry["sig_mc"],
                                   entry["n_hits_total_map"]):
            r["event_id"] = entry["event_id"]
            r["seed"] = entry["seed"]
            track_records.append(r)
        for r in per_cluster_records(labels, entry["sig_mc"]):
            r["event_id"] = entry["event_id"]
            r["seed"] = entry["seed"]
            cluster_records.append(r)
        if log_every and (i + 1) % log_every == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"    {i+1}/{n_evt}  {rate:.1f} ev/s  "
                  f"ETA {(n_evt - i - 1)/max(rate, 1e-3)/60:.1f} min")
    return track_records, cluster_records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_path", default="eval_results/v35_forward_cache")
    p.add_argument("--data_dir",
                   default="/home/marko.cechovic/cgatr/data_parquet_train")
    p.add_argument("--out_dir", default="eval_results/v35_user_greedy_sweep")
    p.add_argument("--max_events", type=int, default=None)
    p.add_argument(
        "--tbetas", nargs="+", type=float,
        default=[0.05, 0.10, 0.20, 0.30, 0.50, 0.70],
    )
    p.add_argument(
        "--tds", nargs="+", type=float,
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50],
    )
    p.add_argument("--save_caches_for", nargs="+", default=[],
                   help="List of 'tbeta:td' OPs whose full FCC cache to materialise "
                        "(e.g. 0.10:0.20 0.70:0.05). Files go to "
                        "eval_results/v35_fcc_user_tbeta{TB}_td{TD}/.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cache from {args.cache_path}")
    hits_df = pl.read_parquet(Path(args.cache_path) / "forward_hits.parquet")
    with open(Path(args.cache_path) / "manifest.json") as f:
        manifest = json.load(f)
    embed_dim = manifest["embed_dim"]
    print(f"  embed_dim={embed_dim} (user greedy is dimension-agnostic)")

    cache = cache_from_dataframe(hits_df, embed_dim)
    if args.max_events is not None:
        cache = cache[: args.max_events]
    n_evt = len(cache)
    n_hits = sum(int(e["sig_coords"].shape[0]) for e in cache)
    print(f"  {n_evt} events, {n_hits} signal hits "
          f"(median {n_hits / n_evt:.0f}/event)")

    seed_lo, seed_hi = manifest["seeds"].split("-")
    seed_lo, seed_hi = int(seed_lo), int(seed_hi) + 1
    print(f"Loading mc_particles for seeds {seed_lo}-{seed_hi - 1}")
    mc_df = load_mc_particles(args.data_dir, seed_lo, seed_hi)

    save_targets = set()
    for s in args.save_caches_for:
        tb, td = s.split(":")
        save_targets.add((float(tb), float(td)))

    pt_bins = [0.10, 0.15, 0.20, 0.30, 0.50, 1.00, 3.00, 30.0]
    overall_rows: list[dict] = []
    per_pt_rows: list[dict] = []

    print(f"\nSweep grid: {len(args.tbetas)} x {len(args.tds)} = "
          f"{len(args.tbetas) * len(args.tds)} combinations")

    for tbeta, td in product(args.tbetas, args.tds):
        print(f"\n=== tbeta={tbeta}, td={td} ===")
        t0 = time.time()
        track_records, cluster_records = run_op(cache, tbeta, td)
        joined, cluster_joined = join_with_mc(
            track_records, cluster_records, mc_df,
        )
        summary = overall_summary(joined)
        summary["fake_rate_idea"] = fake_rate_idea(joined, cluster_joined)
        summary["tbeta"] = tbeta
        summary["td"] = td
        summary["n_clusters"] = len(cluster_records)
        summary["wall_seconds"] = round(time.time() - t0, 1)
        overall_rows.append(summary)
        print(f"  loose={summary['loose_idea']*100:.2f}%  "
              f"strict50={summary['strict50_idea']*100:.2f}%  "
              f"strict90={summary['strict90_idea']*100:.2f}%  "
              f"strict99={summary['strict99_idea']*100:.2f}%  "
              f"fake={summary['fake_rate_idea']*100:.2f}%  "
              f"({summary['wall_seconds']:.0f}s)")

        for r in per_pt_match_rate(joined, pt_bins, eff_thresh=0.50):
            per_pt_rows.append({
                "tbeta": tbeta, "td": td,
                "pt_lo": r["pt_lo"], "pt_hi": r["pt_hi"],
                "n": r["n"],
                "loose": r["match_rate"],
                "strict50": r["match_rate_strict"],
            })

        if (tbeta, td) in save_targets:
            fcc_dir = Path(
                f"eval_results/v35_fcc_user_tbeta{tbeta}_td{td}"
            )
            fcc_dir.mkdir(parents=True, exist_ok=True)
            joined.write_parquet(fcc_dir / "cache.parquet",
                                 compression="zstd")
            cluster_joined.write_parquet(fcc_dir / "cache_clusters.parquet",
                                         compression="zstd")
            (fcc_dir / "manifest.json").write_text(json.dumps({
                "source_cache": args.cache_path,
                "tbeta": tbeta, "td": td,
                "algorithm": "user_greedy_self_seed",
                "n_events": n_evt,
                "n_tracks": joined.height,
                "n_clusters": cluster_joined.height,
            }, indent=2))
            print(f"  Saved FCC cache to {fcc_dir}/")

    overall_df = pl.DataFrame(overall_rows)
    overall_df.write_csv(out_dir / "sweep.csv")
    pl.DataFrame(per_pt_rows).write_csv(out_dir / "per_pt.csv")
    print(f"\nWrote {out_dir / 'sweep.csv'}")

    md = ["# User-greedy parameter sweep (v35 cache, IDEA cut)\n",
          "",
          "Self-seed greedy on the v35 5-D embedding. tbeta is the "
          "minimum sigmoid(beta) for a hit to seed a cluster; td is the "
          "radius (in embedding units) that absorbs unassigned hits.\n",
          ""]
    md.append("| tbeta | td | n_clusters | loose | strict50 | strict75 | strict90 | strict99 | fake | wall_s |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in overall_rows:
        md.append(
            f"| {r['tbeta']:.2f} | {r['td']:.2f} | {r['n_clusters']} | "
            f"{r['loose_idea']*100:.2f}% | {r['strict50_idea']*100:.2f}% | "
            f"{r['strict75_idea']*100:.2f}% | {r['strict90_idea']*100:.2f}% | "
            f"{r['strict99_idea']*100:.2f}% | "
            f"{r['fake_rate_idea']*100:.2f}% | {r['wall_seconds']:.0f} |"
        )
    (out_dir / "sweep.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {out_dir / 'sweep.md'}")

    metric_keys = [
        ("loose_idea", "Loose (purity \u2265 0.75)"),
        ("strict50_idea", "Strict50 (eff \u2265 0.50)"),
        ("strict75_idea", "Strict75 (eff \u2265 0.75)"),
        ("strict90_idea", "Strict90 (eff \u2265 0.90)"),
        ("strict99_idea", "Strict99 (eff \u2265 0.99)"),
        ("fake_rate_idea", "Fake rate"),
    ]
    n_tb = len(args.tbetas)
    n_td = len(args.tds)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (key, title) in zip(axes.flat, metric_keys):
        grid = np.full((n_tb, n_td), np.nan)
        for r in overall_rows:
            i = args.tbetas.index(r["tbeta"])
            j = args.tds.index(r["td"])
            grid[i, j] = r[key]
        if key == "fake_rate_idea":
            im = ax.imshow(grid, origin="lower", aspect="auto",
                           cmap="Reds", vmin=0.0,
                           vmax=float(np.nanmax(grid)))
        else:
            im = ax.imshow(grid, origin="lower", aspect="auto",
                           cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(n_td))
        ax.set_xticklabels([f"{t:.2f}" for t in args.tds])
        ax.set_yticks(range(n_tb))
        ax.set_yticklabels([f"{t:.2f}" for t in args.tbetas])
        ax.set_xlabel("td")
        ax.set_ylabel("tbeta")
        ax.set_title(title)
        for i in range(n_tb):
            for j in range(n_td):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]*100:.0f}",
                            ha="center", va="center",
                            color="white" if grid[i, j] < 0.5 else "black",
                            fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(
        "User-greedy (self-seed) sweep on v35 forward cache — IDEA cut",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "heatmaps.png", dpi=130)
    plt.close(fig)
    print(f"Wrote {out_dir / 'heatmaps.png'}")


if __name__ == "__main__":
    main()
