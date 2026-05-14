"""Per-event ONNX parity check + minimal inference example.

For each event in a parquet seed range:
  1. Wrap the event as a `B == 1` batch (features `[1, N, 10]`,
     padding_mask `[1, N]` all-True).
  2. Run it through the exported ONNX (`coords_and_beta [1, N, F]`).
  3. Run the same `OnnxV35` torch wrapper on CPU on the same input.
  4. Record `max_abs_diff(embed)`, `max_abs_diff(beta)`, and per-
     event torch / ORT wall-clock.
  5. Write `<out>/per_event_parity.csv` and `<out>/parity.md`.

Acceptance (rule of thumb): both diffs <= 1e-4 means the C++ deployment
inference will see numerically identical features to the training-time
PyTorch model. Typical observed: ~5e-7 (numerical noise).

For a full FCC-metric eval using the exported ONNX (clustering +
matching), see `eval_fcc_metrics_v35_ort.py` in the same folder.

Usage (from repo root):
    PYTHONPATH=model_training python conversion_to_onnx/onnx_parity_v35.py \\
        --checkpoint <ckpt> --onnx <onnx> --embed_dim 4 \\
        --data_dir /path/to/parquet --val_seeds 181-181 \\
        --max_events 50 --max_hits 5000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "model_training"))

# onnx_export_v35 lives next to this script (conversion_to_onnx/), so
# import it via its package path; OnnxV35/load_v35_weights/wrap_single_event
# are the public surface for parity checks.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_onnx_export_v35",
                                     os.path.join(_THIS, "onnx_export_v35.py"))
_onnx_export = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_onnx_export)
OnnxV35 = _onnx_export.OnnxV35
load_v35_weights = _onnx_export.load_v35_weights
wrap_single_event = _onnx_export.wrap_single_event
from src.dataset.parquet_dataset import IDEAParquetDataset


def parse_seed_range(s):
    a, b = s.split("-")
    return int(a), int(b) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="checkpoints/v35_smoke/cgatr_best.pt")
    ap.add_argument("--onnx",
                    default="checkpoints/v35_smoke/cgatr_v35_smoke.onnx")
    ap.add_argument("--data_dir",
                    default="/home/marko.cechovic/cgatr/data_parquet_train")
    ap.add_argument("--val_seeds", default="181-181")
    ap.add_argument("--max_events", type=int, default=200)
    ap.add_argument("--max_hits", type=int, default=3000)
    ap.add_argument("--embed_dim", type=int, default=4,
                    help="Must match the embed_dim of the trained model "
                         "behind --checkpoint / --onnx. v35-Lightning "
                         "defaults to 4; legacy v35 was 5.")
    ap.add_argument("--num_blocks", type=int, default=10)
    ap.add_argument("--out_dir",
                    default="eval_results/v35_onnx_smoke")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import onnxruntime as ort

    print(f"[onnx_parity_v35] loading ckpt: {args.checkpoint}")
    model = OnnxV35(num_blocks=args.num_blocks, hidden_mv_channels=16,
                    hidden_s_channels=64, embed_dim=args.embed_dim)
    load_v35_weights(model, args.checkpoint)
    model.eval()

    print(f"[onnx_parity_v35] loading ONNX: {args.onnx}")
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    seed_start, seed_end = parse_seed_range(args.val_seeds)
    dataset = IDEAParquetDataset(args.data_dir, seed_range=(seed_start, seed_end),
                                 max_hits_per_event=args.max_hits)
    n_total = min(args.max_events, len(dataset))
    print(f"[onnx_parity_v35] dataset: {len(dataset)} events "
          f"(processing {n_total})")

    rows = []
    skipped = 0
    t_total_t0 = time.time()
    for idx in range(n_total):
        event = dataset[idx]
        if event is None:
            skipped += 1
            continue

        features = event["features"].cpu()
        n_hits = int(event["n_hits"])
        if n_hits < 2:
            skipped += 1
            continue

        # Both backends speak the same batched API; wrap as B=1 so
        # the call sites match production multi-event inference.
        feats_b, pad_b = wrap_single_event(features)

        # Torch CPU forward
        t0 = time.perf_counter()
        with torch.no_grad():
            y_torch_b = model(feats_b, pad_b)                  # [1, N, F]
        t_torch_ms = (time.perf_counter() - t0) * 1000.0

        # ORT CPU forward
        t0 = time.perf_counter()
        y_ort_b = sess.run(None, {
            "features": feats_b.numpy(),
            "padding_mask": pad_b.numpy(),
        })[0]                                                    # [1, N, F]
        t_ort_ms = (time.perf_counter() - t0) * 1000.0

        # Strip the trivial batch dim for parity comparison.
        y_torch_np = y_torch_b[0].numpy()
        y_ort = y_ort_b[0]

        ed = args.embed_dim
        embed_diff = float(np.max(np.abs(
            y_torch_np[:, :ed] - y_ort[:, :ed]
        )))
        beta_diff = float(np.max(np.abs(
            y_torch_np[:, ed] - y_ort[:, ed]
        )))
        rows.append({
            "event_idx": idx,
            "n_hits": n_hits,
            "max_abs_diff_embed": embed_diff,
            "max_abs_diff_beta": beta_diff,
            "torch_ms": t_torch_ms,
            "ort_ms": t_ort_ms,
        })

        if (idx + 1) % 25 == 0 or idx + 1 == n_total:
            avg_torch = np.mean([r["torch_ms"] for r in rows])
            avg_ort = np.mean([r["ort_ms"] for r in rows])
            max_embed = max(r["max_abs_diff_embed"] for r in rows)
            max_beta = max(r["max_abs_diff_beta"] for r in rows)
            print(
                f"  Event {idx + 1}/{n_total}  "
                f"max|d_embed|={max_embed:.2e}  max|d_beta|={max_beta:.2e}  "
                f"torch={avg_torch:.0f}ms ort={avg_ort:.0f}ms",
                flush=True,
            )

    total_wall = time.time() - t_total_t0

    # ---- write CSV --------------------------------------------------------
    csv_path = out / "per_event_parity.csv"
    with csv_path.open("w") as f:
        f.write("event_idx,n_hits,max_abs_diff_embed,max_abs_diff_beta,"
                "torch_ms,ort_ms\n")
        for r in rows:
            f.write(
                f"{r['event_idx']},{r['n_hits']},"
                f"{r['max_abs_diff_embed']:.6e},"
                f"{r['max_abs_diff_beta']:.6e},"
                f"{r['torch_ms']:.3f},{r['ort_ms']:.3f}\n"
            )
    print(f"Wrote {csv_path}")

    # ---- summary md -------------------------------------------------------
    embed_max = max(r["max_abs_diff_embed"] for r in rows) if rows else 0.0
    beta_max = max(r["max_abs_diff_beta"] for r in rows) if rows else 0.0
    embed_mean = (
        np.mean([r["max_abs_diff_embed"] for r in rows]) if rows else 0.0
    )
    beta_mean = (
        np.mean([r["max_abs_diff_beta"] for r in rows]) if rows else 0.0
    )
    torch_mean_ms = np.mean([r["torch_ms"] for r in rows]) if rows else 0.0
    ort_mean_ms = np.mean([r["ort_ms"] for r in rows]) if rows else 0.0
    embed_gate = embed_max <= args.tol
    beta_gate = beta_max <= args.tol

    lines = [
        "# ONNX per-event parity (smoke val seed)\n",
        f"- checkpoint: `{args.checkpoint}`",
        f"- onnx:       `{args.onnx}`",
        f"- val seeds:  {args.val_seeds}  ({len(rows)} events processed, "
        f"{skipped} skipped)",
        f"- total wall: {total_wall:.1f}s",
        "",
        "## Per-event parity",
        f"- max |torch - ort| embed: **{embed_max:.3e}**  "
        f"(tol {args.tol:.0e}) -> {'PASS' if embed_gate else 'FAIL'}",
        f"- max |torch - ort| beta:  **{beta_max:.3e}**  "
        f"(tol {args.tol:.0e}) -> {'PASS' if beta_gate else 'FAIL'}",
        f"- mean |diff| embed:       {embed_mean:.3e}",
        f"- mean |diff| beta:        {beta_mean:.3e}",
        "",
        "## Per-event timing (CPU)",
        f"- torch CPU mean: {torch_mean_ms:.1f} ms/event",
        f"- ORT CPU mean:   {ort_mean_ms:.1f} ms/event",
        f"- ORT / torch ratio: {ort_mean_ms / max(torch_mean_ms, 1e-6):.2f}x",
        "",
        f"**Overall: {'PASS' if (embed_gate and beta_gate) else 'FAIL'}**\n",
    ]
    (out / "parity.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out / 'parity.md'}")

    # ---- JSON for downstream scripts -------------------------------------
    (out / "parity.json").write_text(json.dumps({
        "n_events": len(rows),
        "n_skipped": skipped,
        "embed_max": embed_max,
        "beta_max": beta_max,
        "embed_mean": embed_mean,
        "beta_mean": beta_mean,
        "torch_mean_ms": torch_mean_ms,
        "ort_mean_ms": ort_mean_ms,
        "total_wall_s": total_wall,
        "tol": args.tol,
        "pass": bool(embed_gate and beta_gate),
    }, indent=2))


if __name__ == "__main__":
    main()
