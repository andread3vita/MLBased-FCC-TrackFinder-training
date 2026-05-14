"""FCC-style tracking metrics using ONNX Runtime for the forward pass.

Mirrors `model_training/src/eval_user_greedy_sweep.py` but swaps the
torch forward for an `onnxruntime.InferenceSession.run(...)` call
against the batched ONNX export. Each event is wrapped as a `B == 1`
batch (features `[1, N, 10]`, padding_mask `[1, N]` all-True). The
clustering (`user_greedy.get_clustering_user`), per-track /
per-cluster bookkeeping, mc-particle join, and reconstructable-mask
logic are reused verbatim from `eval_fcc_metrics_v36.py` so the
comparison vs the torch eval is purely "same numbers, different
forward backend".

Output: `<output_dir>/cache.parquet, cache_clusters.parquet,
summary.json` (same schema as the torch eval, which makes side-by-side
comparison trivial).

Usage (from repo root):
    PYTHONPATH=model_training python conversion_to_onnx/eval_fcc_metrics_v35_ort.py \\
        --data_dir /path/to/parquet_data \\
        --onnx checkpoints/cgatr_v35.onnx \\
        --eval_seeds 181-181 --max_events 50 \\
        --output_dir eval_results/v35_onnx_fcc
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "model_training"))

from src.user_greedy import get_clustering_user
from src.dataset.parquet_dataset import IDEAParquetDataset


def parse_seed_range(s: str) -> tuple[int, int]:
    """Parse '181-190' as Python half-open range (181, 191)."""
    a, b = s.split("-")
    return int(a), int(b) + 1


from src.eval_fcc_metrics_v36 import (
    per_track_records,
    per_cluster_records,
    load_mc_particles,
    add_reconstructable_masks,
)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def run_ort_inference(sess, dataset, max_events, embed_dim, tbeta, td,
                      log_every=20):
    """Mirrors `eval_fcc_metrics_v36.run_inference` but uses ORT."""
    track_records = []
    cluster_records = []
    forward_times_ms = []

    n_events = min(max_events, len(dataset))
    n_skipped = 0
    t0 = time.time()

    for idx in range(n_events):
        event = dataset[idx]
        if event is None:
            n_skipped += 1
            continue

        features_np = event["features"].cpu().numpy().astype(np.float32)
        mc_index_all = event["mc_index"].numpy()
        is_secondary = event["is_secondary"].numpy().astype(bool)

        sig_mask = (~is_secondary) & (mc_index_all != 0)
        if sig_mask.sum() < 4:
            n_skipped += 1
            continue

        # Wrap as B=1 (features [1, N, 10], padding_mask [1, N] all True).
        n_hits = features_np.shape[0]
        feats_b = features_np[None, ...]
        pad_b = np.ones((1, n_hits), dtype=bool)

        t_fwd = time.perf_counter()
        output_b = sess.run(None, {
            "features": feats_b,
            "padding_mask": pad_b,
        })[0]
        forward_times_ms.append((time.perf_counter() - t_fwd) * 1000.0)

        # Strip the trivial batch dim for downstream per-event work.
        output = output_b[0]                                   # [N, F]
        coords = output[:, :embed_dim]
        beta = sigmoid_np(output[:, embed_dim])

        sig_coords = coords[sig_mask]
        sig_beta = beta[sig_mask]
        sig_mc = mc_index_all[sig_mask]

        n_hits_total_map = {}
        unique_mc_all = np.unique(mc_index_all)
        unique_mc_all = unique_mc_all[unique_mc_all > 0]
        for mc_idx in unique_mc_all:
            n_hits_total_map[int(mc_idx)] = int((mc_index_all == mc_idx).sum())

        labels = get_clustering_user(sig_beta, sig_coords,
                                     tbeta=tbeta, td=td)

        track_rows = per_track_records(labels, sig_mc, n_hits_total_map)
        cluster_rows = per_cluster_records(labels, sig_mc)

        dc_path, vtx_path, eid, _ = dataset._index[idx]
        seed = int(Path(dc_path).parent.name.replace("seed_", ""))
        for tr in track_rows:
            tr["event_id"] = int(eid)
            tr["seed"] = int(seed)
            track_records.append(tr)
        for cr in cluster_rows:
            cr["event_id"] = int(eid)
            cr["seed"] = int(seed)
            cluster_records.append(cr)

        if (idx + 1) % log_every == 0 or idx + 1 == n_events:
            dt = time.time() - t0
            rate = (idx + 1) / max(dt, 1e-3)
            avg_fwd = np.mean(forward_times_ms) if forward_times_ms else 0
            print(
                f"  Event {idx + 1}/{n_events}  "
                f"({len(track_records)} tracks, {len(cluster_records)} clusters)  "
                f"{rate:.2f} ev/s  ORT fwd avg {avg_fwd:.0f} ms  "
                f"skipped={n_skipped}",
                flush=True,
            )

    print(
        f"Inference done in {(time.time() - t0) / 60.0:.1f} min "
        f"({len(track_records)} tracks, {len(cluster_records)} clusters, "
        f"{n_skipped} events skipped)"
    )
    return track_records, cluster_records, forward_times_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--eval_seeds", default="181-181")
    ap.add_argument("--max_events", type=int, default=50)
    ap.add_argument("--max_hits", type=int, default=1000)
    ap.add_argument("--embed_dim", type=int, default=4,
                    help="Must match the embed_dim of the model behind "
                         "--onnx. v35-Lightning defaults to 4; legacy v35 "
                         "was 5.")
    ap.add_argument("--tbeta", type=float, default=0.1)
    ap.add_argument("--td", type=float, default=0.2)
    ap.add_argument("--output_dir", default="eval_results/v35_onnx_smoke_fcc")
    ap.add_argument("--tag", default="v35_onnx_smoke")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    import onnxruntime as ort
    print(f"Loading ONNX: {args.onnx}")
    sess = ort.InferenceSession(args.onnx,
                                providers=["CPUExecutionProvider"])

    seed_start, seed_end = parse_seed_range(args.eval_seeds)
    print(f"Loading eval data: seeds {seed_start}-{seed_end - 1}, "
          f"max_hits={args.max_hits}")
    dataset = IDEAParquetDataset(args.data_dir, seed_range=(seed_start, seed_end),
                                 max_hits_per_event=args.max_hits)
    print(f"Dataset: {len(dataset)} events (processing {args.max_events})")

    print(f"\n=== ORT inference + greedy (tbeta={args.tbeta}, td={args.td}) ===")
    t_eval_start = time.time()
    tracks, clusters, fwd_ms = run_ort_inference(
        sess, dataset, args.max_events,
        embed_dim=args.embed_dim, tbeta=args.tbeta, td=args.td,
    )
    eval_wall = time.time() - t_eval_start
    if not tracks:
        print("No tracks produced. Exiting.")
        return

    print(f"\nLoading mc_particles for seeds {seed_start}-{seed_end - 1}...")
    mc_df = load_mc_particles(args.data_dir, seed_start, seed_end)
    if mc_df is None:
        print("ERROR: No mc_particles found")
        return

    track_pl = pl.DataFrame(tracks)
    joined = track_pl.join(
        mc_df,
        left_on=["mc_idx", "event_id", "seed"],
        right_on=["mc_index", "event_id", "seed"],
        how="left",
    )
    joined = joined.filter(pl.col("pt").is_not_null())
    joined = add_reconstructable_masks(joined)

    cluster_pl = pl.DataFrame(clusters)
    cluster_joined = cluster_pl.join(
        mc_df.select(["mc_index", "event_id", "seed", "pt", "theta", "phi",
                      "gen_status", "charge", "decayed_in_tracker"]),
        left_on=["matched_mc_idx", "event_id", "seed"],
        right_on=["mc_index", "event_id", "seed"],
        how="left",
    )

    reco_idea_set = set(zip(
        joined.filter(pl.col("is_reconstructable_idea"))["mc_idx"].to_list(),
        joined.filter(pl.col("is_reconstructable_idea"))["event_id"].to_list(),
        joined.filter(pl.col("is_reconstructable_idea"))["seed"].to_list(),
    ))
    reco_cld_set = set(zip(
        joined.filter(pl.col("is_reconstructable_cld"))["mc_idx"].to_list(),
        joined.filter(pl.col("is_reconstructable_cld"))["event_id"].to_list(),
        joined.filter(pl.col("is_reconstructable_cld"))["seed"].to_list(),
    ))

    purs = cluster_joined["purity"].to_numpy()
    cl_mc = cluster_joined["matched_mc_idx"].to_numpy()
    cl_ev = cluster_joined["event_id"].to_numpy()
    cl_sd = cluster_joined["seed"].to_numpy()

    def fake_array(reco_set):
        out = np.empty(len(purs), dtype=bool)
        for i in range(len(purs)):
            if purs[i] < 0.75:
                out[i] = True
            else:
                out[i] = (int(cl_mc[i]), int(cl_ev[i]), int(cl_sd[i])) not in reco_set
        return out

    cluster_joined = cluster_joined.with_columns([
        pl.Series("is_fake_idea", fake_array(reco_idea_set)),
        pl.Series("is_fake_cld", fake_array(reco_cld_set)),
    ])

    out_tracks = os.path.join(args.output_dir, "cache.parquet")
    out_clusters = os.path.join(args.output_dir, "cache_clusters.parquet")
    joined.write_parquet(out_tracks)
    cluster_joined.write_parquet(out_clusters)

    def overall(df: pl.DataFrame, mask_col: str | None):
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

    summary = {
        "onnx": args.onnx,
        "eval_seeds": args.eval_seeds,
        "max_events": args.max_events,
        "max_hits": args.max_hits,
        "tbeta": args.tbeta,
        "td": args.td,
        "n_tracks_total": len(joined),
        "n_clusters_total": len(cluster_joined),
        "no_cuts": overall(joined, None),
        "idea": overall(joined, "is_reconstructable_idea"),
        "cld": overall(joined, "is_reconstructable_cld"),
        "displaced": overall(joined, "is_reconstructable_displaced"),
        "fake_rate_idea":
            float(cluster_joined["is_fake_idea"].sum()) / max(len(cluster_joined), 1),
        "fake_rate_cld":
            float(cluster_joined["is_fake_cld"].sum()) / max(len(cluster_joined), 1),
        "overall_purity": float(cluster_joined["purity"].mean()),
        "eval_wall_s": eval_wall,
        "ort_forward_mean_ms": float(np.mean(fwd_ms)) if fwd_ms else 0.0,
    }

    print("\n" + "=" * 70)
    print(f"FCC-style summary  ({args.tag})  (ORT CPU)")
    print("=" * 70)
    print(f"tracks={summary['n_tracks_total']}  "
          f"clusters={summary['n_clusters_total']}")
    print(f"  overall purity:        {summary['overall_purity']:.3f}")
    print(f"  fake rate (IDEA):      {summary['fake_rate_idea']:.3f}")
    print(f"  fake rate (CLD):       {summary['fake_rate_cld']:.3f}")
    for tag in ["no_cuts", "idea", "cld", "displaced"]:
        s = summary[tag]
        print(f"  [{tag:>9}] n={s['n']:>6}  eff_hit={s['efficiency']:.3f}  "
              f"match_rate={s['match_rate']:.3f}")
    print(f"  ORT fwd mean: {summary['ort_forward_mean_ms']:.0f} ms/event")
    print(f"  eval wall:    {summary['eval_wall_s']:.1f} s")

    out_summary = os.path.join(args.output_dir, "summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_summary}")


if __name__ == "__main__":
    main()
