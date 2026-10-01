"""CPU-only health audit for a live corrected C-GATr checkpoint.

The script snapshots the mutable best-checkpoint path, loads the EMA weights
exactly like ``forward_pass.py``, and forwards a small set of naturally sized,
uncropped validation events. It never exposes CUDA, so it is safe to run while
the production DDP job owns all training GPUs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from src.dataset.parquet_dataset import IDEAParquetDataset
from src.eval.analyze_cached_loss_components import _event_diagnostics
from src.eval.forward_pass import (
    _load_state_dict_tolerating_derived_buffers,
    _make_args,
    _validate_checkpoint_config,
)
from src.model import CGATrParquetModel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_args() -> SimpleNamespace:
    return SimpleNamespace(
        num_blocks=10,
        hidden_mv_channels=16,
        hidden_s_channels=64,
        embed_dim=4,
        normalize_mv_inputs=False,
        algebra="conformal",
        use_time=False,
        pga_hit_encoding="line",
        two_channel_dc=False,
        cga_hit_encoding="sphere_circle",
        physical_drift_geometry=True,
        separate_hit_metadata=False,
        no_legacy_equivariance=True,
        equivariance_group="e3",
        invariant_output_head=True,
        fix_cga_null=True,
        fix_wire_dir=True,
        equi_init="identity_algebra",
    )


def _build_model(checkpoint: dict) -> tuple[CGATrParquetModel, dict]:
    expected = _expected_args()
    _validate_checkpoint_config(checkpoint, expected)
    args = _make_args(
        num_blocks=expected.num_blocks,
        embed_dim=expected.embed_dim,
        hidden_mv_channels=expected.hidden_mv_channels,
        hidden_s_channels=expected.hidden_s_channels,
        legacy_equivariance=False,
        equivariance_group=expected.equivariance_group,
        invariant_output_head=expected.invariant_output_head,
        algebra=expected.algebra,
        use_time=expected.use_time,
        pga_hit_encoding=expected.pga_hit_encoding,
        two_channel_dc=expected.two_channel_dc,
        cga_hit_encoding=expected.cga_hit_encoding,
        physical_drift_geometry=expected.physical_drift_geometry,
        separate_hit_metadata=expected.separate_hit_metadata,
        equi_init=expected.equi_init,
        fix_cga_null=expected.fix_cga_null,
        fix_wire_dir=expected.fix_wire_dir,
        normalize_mv_inputs=expected.normalize_mv_inputs,
    )
    model = CGATrParquetModel(args)
    if checkpoint.get("ema_state_dict") is None:
        raise RuntimeError("checkpoint has no EMA state; refusing a non-validation-equivalent audit")
    state = checkpoint["ema_state_dict"]
    _load_state_dict_tolerating_derived_buffers(model, state)
    bad = [
        name for name, tensor in state.items()
        if torch.is_tensor(tensor) and not torch.isfinite(tensor).all()
    ]
    if bad:
        raise RuntimeError(f"non-finite EMA tensors: {bad[:10]}")
    model.eval()
    return model, state


def _choose_events(
    dataset: IDEAParquetDataset, target_sizes: list[int]
) -> list[tuple[int, int]]:
    available = [
        (index, int(entry[3])) for index, entry in enumerate(dataset._index)
    ]
    chosen: list[tuple[int, int]] = []
    used: set[int] = set()
    for target in target_sizes:
        index, size = min(
            (item for item in available if item[0] not in used),
            key=lambda item: abs(item[1] - target),
        )
        used.add(index)
        chosen.append((index, size))
    return chosen


def _knn_truth_fraction(coords: np.ndarray, truth: np.ndarray, k: int = 10) -> float:
    if len(coords) < 2:
        return float("nan")
    k = min(k, len(coords) - 1)
    indices = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(
        return_distance=False
    )[:, 1:]
    return float(np.mean(truth[indices] == truth[:, None]))


def _event_geometry(
    coords: np.ndarray, beta: np.ndarray, truth: np.ndarray
) -> dict:
    ids = np.unique(truth)
    alpha_indices = np.asarray([
        np.flatnonzero(truth == target)[np.argmax(beta[truth == target])]
        for target in ids
    ])
    alphas = coords[alpha_indices]
    own_alpha = np.empty_like(coords)
    for target, alpha in zip(ids, alphas):
        own_alpha[truth == target] = alpha
    own_distance = np.linalg.norm(coords - own_alpha, axis=1)

    if len(alphas) > 1:
        nn = NearestNeighbors(n_neighbors=2).fit(alphas)
        alpha_nearest = nn.kneighbors(return_distance=True)[0][:, 1]
    else:
        alpha_nearest = np.asarray([float("nan")])

    alpha_mask = np.zeros(len(coords), dtype=bool)
    alpha_mask[alpha_indices] = True
    nonalpha = ~alpha_mask
    own_median = float(np.median(own_distance))
    nearest_median = float(np.nanmedian(alpha_nearest))
    return {
        "n_targets": int(len(ids)),
        "own_distance_median": own_median,
        "own_distance_p90": float(np.quantile(own_distance, 0.9)),
        "nearest_alpha_distance_median": nearest_median,
        "alpha_margin_collision_fraction": float(np.nanmean(alpha_nearest < 1.0)),
        "separation_over_compactness": nearest_median / max(own_median, 1e-12),
        "knn10_same_track_fraction": _knn_truth_fraction(coords, truth),
        "beta_alpha_median": float(np.median(beta[alpha_mask])),
        "beta_nonalpha_median": (
            float(np.median(beta[nonalpha])) if nonalpha.any() else float("nan")
        ),
        "beta_gt_08_per_target": float((beta > 0.8).sum() / max(len(ids), 1)),
    }


def _read_trajectory(epoch_csv: Path, step_csv: Path) -> dict:
    with epoch_csv.open() as handle:
        epochs = list(csv.DictReader(handle))
    if not epochs:
        raise RuntimeError("no completed epochs")
    numeric = [
        {
            "epoch": int(row["epoch"]),
            "train_loss": float(row["mean_train_loss"]),
            "val_loss": float(row["val_loss"]),
            "strict50": float(row["val_match_strict50"]),
            "loose": float(row["val_match_loose"]),
            "lr": float(row["lr"]),
        }
        for row in epochs
    ]

    steps = pl.read_csv(step_csv, ignore_errors=True)
    component_columns = [
        "train/L_att", "train/L_rep", "train/L_var", "train/L_beta_sig",
        "train/L_beta_noise", "train/L_beta_suppress", "train/loss",
    ]
    available = [column for column in component_columns if column in steps.columns]
    components = (
        steps.filter(pl.col("epoch").is_not_null())
        .group_by("epoch")
        .agg([pl.col(column).drop_nulls().mean().alias(column) for column in available])
        .sort("epoch")
    )
    return {
        "epochs": numeric,
        "best_epoch_by_val_loss": min(numeric, key=lambda row: row["val_loss"])["epoch"],
        "val_loss_strictly_improving": all(
            right["val_loss"] < left["val_loss"]
            for left, right in zip(numeric, numeric[1:])
        ),
        "finite": all(
            math.isfinite(value)
            for row in numeric
            for key, value in row.items()
            if key != "epoch"
        ),
        "component_epoch_means": components.to_dicts(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epoch-csv", required=True)
    parser.add_argument("--step-csv", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--target-sizes", default="500,1000,1500,2000,2500,3000,3750",
        help="Natural (uncropped) validation-event sizes to sample",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        raise RuntimeError("CUDA is visible; rerun with CUDA_VISIBLE_DEVICES= empty")
    torch.set_num_threads(2)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.checkpoint)
    snapshot = output / "checkpoint_snapshot.ckpt"
    size_before = source.stat().st_size
    time.sleep(1.0)
    if source.stat().st_size != size_before:
        raise RuntimeError("checkpoint is changing; retry after the epoch boundary")
    shutil.copy2(source, snapshot)
    snapshot_sha = _sha256(snapshot)

    checkpoint = torch.load(snapshot, map_location="cpu", weights_only=False)
    model, ema_state = _build_model(checkpoint)
    parameter_norm = math.sqrt(sum(
        float(tensor.float().square().sum())
        for name, tensor in ema_state.items()
        if torch.is_tensor(tensor) and name in dict(model.named_parameters())
    ))

    dataset = IDEAParquetDataset(
        args.data_dir,
        seed_range=(181, 191),
        max_hits_per_event=None,
        with_time=False,
        drop_loopers=False,
        merge_daughters=False,
        with_drift_dir=model.needs_drift_dir,
    )
    targets = [int(value) for value in args.target_sizes.split(",")]
    selected = _choose_events(dataset, targets)

    event_reports = []
    cache_frames = []
    with torch.no_grad():
        for index, expected_size in selected:
            event = dataset[index]
            if event is None:
                continue
            started = time.time()
            output_tensor = model(event["features"], [event["n_hits"]])
            if not torch.isfinite(output_tensor).all():
                raise RuntimeError(f"non-finite output for dataset index {index}")
            coords_all = output_tensor[:, :4].numpy().astype(np.float32)
            beta_all = torch.sigmoid(output_tensor[:, 4]).numpy().astype(np.float32)
            mc_all = event["mc_index"].numpy()
            secondary = event["is_secondary"].numpy().astype(bool)
            signal = (~secondary) & (mc_all >= 0)
            ids, counts = np.unique(mc_all[signal], return_counts=True)
            valid_ids = ids[counts >= 3]
            keep = signal & np.isin(mc_all, valid_ids)
            coords = coords_all[keep]
            beta = beta_all[keep]
            truth = mc_all[keep]
            n_hits_by_id = dict(zip(ids.tolist(), counts.tolist()))
            dc_path, _vtx, event_id, _size = dataset._index[index]
            seed = int(Path(dc_path).parent.name.replace("seed_", ""))

            frame = pl.DataFrame({
                "seed": [seed] * len(coords),
                "event_id": [int(event_id)] * len(coords),
                **{f"coord_{dimension}": coords[:, dimension] for dimension in range(4)},
                "beta": beta,
                "mc_index": truth,
                "n_hits_total": [n_hits_by_id[int(value)] for value in truth],
            })
            # The model forward is inference-only, but the loss audit measures
            # gradients with respect to detached output coordinates and beta.
            with torch.enable_grad():
                loss = _event_diagnostics(frame, 4, "paper_hinge", None)
            loss_components = {
                name: value for name, value in loss.items()
                if isinstance(value, dict)
            }
            geometry = _event_geometry(coords, beta, truth)
            event_reports.append({
                "seed": seed,
                "event_id": int(event_id),
                "n_input_hits": int(event["n_hits"]),
                "n_audited_hits": len(coords),
                "forward_seconds": time.time() - started,
                **geometry,
                "loss_components": loss_components,
            })
            cache_frames.append(frame)

    cached = pl.concat(cache_frames)
    x = cached.select([f"coord_{dimension}" for dimension in range(4)]).to_numpy()
    beta = cached["beta"].to_numpy()
    pca = PCA(n_components=4, svd_solver="full").fit(x)
    high = x[beta > 0.8]
    pca_high = PCA(n_components=4, svd_solver="full").fit(high)
    variance = pca.explained_variance_ratio_
    variance_high = pca_high.explained_variance_ratio_

    trajectory = _read_trajectory(Path(args.epoch_csv), Path(args.step_csv))
    report = {
        "checkpoint": {
            "source": str(source.resolve()),
            "snapshot": str(snapshot.resolve()),
            "sha256": snapshot_sha,
            "lightning_epoch_zero_based": int(checkpoint.get("epoch", -1)),
            "global_step": int(checkpoint.get("global_step", -1)),
            "ema_tensor_count": len(ema_state),
            "ema_all_finite": True,
            "ema_parameter_l2_norm": parameter_norm,
        },
        "sample": {
            "policy": "nearest naturally sized uncropped validation event",
            "requested_hit_sizes": targets,
            "events": len(event_reports),
            "hits": int(len(cached)),
            "size_range": [
                min(row["n_input_hits"] for row in event_reports),
                max(row["n_input_hits"] for row in event_reports),
            ],
        },
        "embedding": {
            "coordinate_mean": x.mean(axis=0).tolist(),
            "coordinate_std": x.std(axis=0).tolist(),
            "all_finite": bool(np.isfinite(x).all() and np.isfinite(beta).all()),
            "pca_explained_variance": variance.tolist(),
            "pca_high_beta_explained_variance": variance_high.tolist(),
            "participation_ratio": float(1.0 / np.square(variance).sum()),
            "fourth_component_variance": float(variance[3]),
        },
        "event_reports": event_reports,
        "event_aggregate": {},
        "trajectory": trajectory,
        "limitations": [
            "CPU audit samples seven naturally sized events, not the full validation set.",
            "Maximum sampled occupancy is about 3750 hits; the 15k-hit tail is not forwarded.",
            "Training strict50 uses a fixed operating point; final checkpoint selection uses the later full sweep.",
        ],
    }
    aggregate_keys = [
        "own_distance_median", "own_distance_p90",
        "nearest_alpha_distance_median", "alpha_margin_collision_fraction",
        "separation_over_compactness", "knn10_same_track_fraction",
        "beta_alpha_median", "beta_nonalpha_median", "beta_gt_08_per_target",
    ]
    report["event_aggregate"] = {
        key: float(np.nanmean([row[key] for row in event_reports]))
        for key in aggregate_keys
    }
    component_names = list(event_reports[0]["loss_components"])
    report["event_aggregate"]["loss_components"] = {
        name: {
            metric: float(np.mean([
                row["loss_components"][name][metric] for row in event_reports
            ]))
            for metric in ("value", "coord_gradient_rms", "beta_gradient_rms")
        }
        for name in component_names
    }

    report_path = output / "audit.json"
    report_path.write_text(json.dumps(report, indent=2))
    cached.write_parquet(output / "sampled_embeddings.parquet", compression="zstd")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
