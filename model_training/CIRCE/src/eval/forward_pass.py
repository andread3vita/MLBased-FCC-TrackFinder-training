"""Inference-time forward pass for C-GATr FCC eval pipeline.

Runs the model forward pass on all events in a seed range, caches per-hit
outputs (coords, beta, mc_index) to parquet for downstream clustering.

The disk cache is reusable across runs:
  - First run with --cache_path X writes forward_hits.parquet + manifest.json.
  - Subsequent runs with --cache_path X reuse the cache (no GPU needed).

Usage:
  python src/eval/forward_pass.py \
      --data_dir /path/to/v1_zqq_uds \
      --checkpoint checkpoints/cgatr_fcc_prod/last.ckpt \
      --eval_seeds 1001-1196 \
      --embed_dim 4 --num_blocks 10 \
      --cache_path eval_results/my_run/emb_all
"""

from __future__ import annotations

import argparse as _argparse
import json
import os
import sys as _sys
import time
from pathlib import Path

_sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import polars as pl
import torch

from src.model import CGATrParquetModel
from src.dataset.parquet_dataset import IDEAParquetDataset


def _make_args(num_blocks, embed_dim, hidden_mv_channels=16, hidden_s_channels=64):
    """Build a minimal Namespace to instantiate CGATrParquetModel."""
    ns = _argparse.Namespace(
        hidden_mv_channels=hidden_mv_channels,
        hidden_s_channels=hidden_s_channels,
        num_blocks=num_blocks,
        embed_dim=embed_dim,
        beta_mlp=False,
        normalize_mv_inputs=True,
        cosine_norm=False,
        grad_checkpoint=False,
    )
    return ns


def _parse_seed_range(s: str):
    a, b = s.split("-")
    return int(a), int(b) + 1


def cache_to_dataframe(cache, embed_dim) -> pl.DataFrame:
    """Flatten in-memory cache to a single hit-level polars DataFrame."""
    rows = {"event_id": [], "seed": []}
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
    """Inverse of `cache_to_dataframe`."""
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
def forward_and_cache(model, dataset, device, embed_dim, log_every=200):
    """Run the model forward pass on every event once and cache per-hit outputs."""
    model.eval()
    cache: list[dict] = []
    n_events = len(dataset)
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


def main():
    p = _argparse.ArgumentParser(description="C-GATr FCC forward pass + cache")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval_seeds", default="1001-1196")
    p.add_argument("--max_hits", type=int, default=0,
                   help="Per-event hit cap. 0 = uncapped.")
    p.add_argument("--embed_dim", type=int, default=4)
    p.add_argument("--num_blocks", type=int, default=10)
    p.add_argument("--hidden_mv_channels", type=int, default=16)
    p.add_argument("--hidden_s_channels", type=int, default=64)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--cache_path", type=str, default=None,
        help="Directory for the on-disk forward-pass cache. "
             "Cache contents: forward_hits.parquet + manifest.json.",
    )
    p.add_argument(
        "--force_refresh_cache", action="store_true",
        help="If set with --cache_path, ignore any existing cache and overwrite.",
    )
    args = p.parse_args()

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
            if same_ckpt and same_seeds and same_dim and same_max_hits:
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
                    f"max_hits={same_max_hits}) ==="
                )

    if not cache_used:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        print(f"Loading checkpoint: {args.checkpoint}")
        model_args = _make_args(
            num_blocks=args.num_blocks,
            embed_dim=args.embed_dim,
            hidden_mv_channels=args.hidden_mv_channels,
            hidden_s_channels=args.hidden_s_channels,
        )
        model = CGATrParquetModel(model_args)

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

        seed_start, seed_end = _parse_seed_range(args.eval_seeds)
        max_hits = args.max_hits if args.max_hits > 0 else None
        print(f"Loading eval data: seeds {seed_start}-{seed_end - 1}  max_hits={max_hits}")
        dataset = IDEAParquetDataset(
            args.data_dir,
            seed_range=(seed_start, seed_end),
            max_hits_per_event=max_hits,
        )
        print(f"Dataset: {len(dataset)} events")

        print("\n=== Forward pass + cache ===")
        cache = forward_and_cache(model, dataset, device, args.embed_dim)

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
                    "n_events": len(cache),
                    "n_hits": int(hits_df.height),
                }, f, indent=2)
            print(
                f"  cache written ({hits_df.height} hits, "
                f"{os.path.getsize(os.path.join(args.cache_path, 'forward_hits.parquet')) / 1e6:.1f} MB) "
                f"in {time.time() - t0:.1f}s"
            )
    else:
        print(f"Cache loaded: {len(cache)} events (no GPU forward pass needed)")


if __name__ == "__main__":
    main()
