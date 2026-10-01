"""CPU-only architecture, gradient, radius, optimizer, and trajectory audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.cgatr.layers.layer_norm import EquiLayerNorm
from src.dataset.parquet_dataset import IDEAParquetDataset
from src.eval.audit_checkpoint_embeddings import _build_model
from src.lightning_module import relabel_small_targets
from src.model import object_condensation_loss


def _relative_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    delta = (left.float() - right.float()).square().mean().sqrt()
    scale = left.float().square().mean().sqrt().clamp(min=1e-12)
    return float(delta / scale)


def _choose_event(
    dataset: IDEAParquetDataset,
    target_size: int,
    exclude: set[int] | None = None,
) -> int:
    excluded = exclude or set()
    return min(
        (index for index in range(len(dataset)) if index not in excluded),
        key=lambda index: abs(int(dataset._index[index][3]) - target_size),
    )


def _radius_sensitivity(model, features, seq_lens):
    is_dc = features[:, 3] == 1
    zero = features.clone()
    plus_ten = features.clone()
    zero[is_dc, 7] = 0
    plus_ten[is_dc, 7] *= 1.1

    captures = []
    first_norm = next(
        module for module in model.modules() if isinstance(module, EquiLayerNorm)
    )

    def capture(_module, _inputs, output):
        captures.append(output[0].detach().clone())

    handle = first_norm.register_forward_hook(capture)
    with torch.no_grad():
        base_mv, _ = model.embed(features)
        zero_mv, _ = model.embed(zero)
        ten_mv, _ = model.embed(plus_ten)
        base_output = model(features, seq_lens)
        base_norm = captures[-1]
        repeat_output = model(features, seq_lens)
        repeat_norm = captures[-1]
        zero_output = model(zero, seq_lens)
        zero_norm = captures[-1]
        ten_output = model(plus_ten, seq_lens)
        ten_norm = captures[-1]
    handle.remove()

    radii = features[is_dc, 7]
    return {
        "n_dc_hits": int(is_dc.sum()),
        "raw_radius_mm": {
            "minimum": float(radii.min()),
            "median": float(radii.median()),
            "maximum": float(radii.max()),
            "physical_geometry_scale": 1000.0,
            "scaled_median": float((radii / 1000.0).median()),
        },
        "embedding_relative_rms_zero_radius": _relative_rms(
            base_mv[is_dc], zero_mv[is_dc]
        ),
        "embedding_relative_rms_plus_10_percent": _relative_rms(
            base_mv[is_dc], ten_mv[is_dc]
        ),
        "first_layernorm_relative_rms_zero_radius": _relative_rms(
            base_norm[is_dc], zero_norm[is_dc]
        ),
        "first_layernorm_repeatability_relative_rms": _relative_rms(
            base_norm[is_dc], repeat_norm[is_dc]
        ),
        "first_layernorm_relative_rms_plus_10_percent": _relative_rms(
            base_norm[is_dc], ten_norm[is_dc]
        ),
        "checkpoint_output_relative_rms_zero_radius": _relative_rms(
            base_output[is_dc], zero_output[is_dc]
        ),
        "checkpoint_output_repeatability_relative_rms": _relative_rms(
            base_output[is_dc], repeat_output[is_dc]
        ),
        "checkpoint_output_relative_rms_plus_10_percent": _relative_rms(
            base_output[is_dc], ten_output[is_dc]
        ),
        "coordinate_output_relative_rms_zero_radius": _relative_rms(
            base_output[is_dc, :4], zero_output[is_dc, :4]
        ),
        "beta_logit_output_relative_rms_zero_radius": _relative_rms(
            base_output[is_dc, 4], zero_output[is_dc, 4]
        ),
        "normalization_configuration": {
            "normalize_mv_inputs": bool(model.args.normalize_mv_inputs),
            "separate_hit_metadata": bool(model.separate_hit_metadata),
            "radius_scalar_channel_present": bool(
                model.separate_hit_metadata
                or model.projective
                and model.pga_encoding in ("line", "point_line")
            ),
        },
    }


def _gradient_inventory(model, event):
    model.zero_grad(set_to_none=True)
    output = model(event["features"], [event["n_hits"]])
    coords = output[:, :4].float()
    beta = torch.sigmoid(output[:, 4].float())
    batch_ids = torch.zeros(len(coords), dtype=torch.long)
    mc = event["mc_index"].clone()
    mc[event["is_secondary"]] = -1
    mc = relabel_small_targets(mc, batch_ids, -1, 3)
    loss, components = object_condensation_loss(
        coords, beta, mc, batch_ids, noise_index=-1, qmin=0.1,
        attr_weight=1.0, repul_weight=1.0, beta_suppress_weight=0.1,
        var_weight=0.3, return_components=True, oc_mode="paper_hinge",
    )
    loss.backward()

    rows = []
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        rows.append({
            "name": name,
            "parameters": parameter.numel(),
            "requires_grad": parameter.requires_grad,
            "gradient_present": grad is not None,
            "gradient_finite": bool(
                grad is not None and torch.isfinite(grad).all()
            ),
            "gradient_nonzero": bool(
                grad is not None and torch.count_nonzero(grad) > 0
            ),
            "gradient_rms": (
                float(grad.float().square().mean().sqrt())
                if grad is not None else 0.0
            ),
        })
    used_parameters = sum(
        row["parameters"] for row in rows if row["gradient_nonzero"]
    )
    total_parameters = sum(row["parameters"] for row in rows)
    return {
        "loss": float(loss.detach()),
        "components": {
            key: float(value.detach()) for key, value in components.items()
        },
        "tensor_count": len(rows),
        "parameter_count": total_parameters,
        "nonzero_gradient_parameter_count": used_parameters,
        "nonzero_gradient_fraction": used_parameters / max(total_parameters, 1),
        "missing_gradient_tensors": [
            row["name"] for row in rows if not row["gradient_present"]
        ],
        "zero_gradient_tensors": [
            row["name"] for row in rows
            if row["gradient_present"] and not row["gradient_nonzero"]
        ],
        "nonfinite_gradient_tensors": [
            row["name"] for row in rows
            if row["gradient_present"] and not row["gradient_finite"]
        ],
        "per_tensor": rows,
    }


def _checkpoint_state(checkpoint, model):
    raw = checkpoint.get("state_dict", {})
    ema = checkpoint.get("ema_state_dict", {})
    squared_delta = 0.0
    squared_ema = 0.0
    compared = 0
    for name in dict(model.named_parameters()):
        raw_tensor = raw.get(f"model.{name}")
        ema_tensor = ema.get(name)
        if raw_tensor is None or ema_tensor is None:
            continue
        squared_delta += float((raw_tensor.float() - ema_tensor.float()).square().sum())
        squared_ema += float(ema_tensor.float().square().sum())
        compared += raw_tensor.numel()

    optimizer = checkpoint.get("optimizer_states", [])
    optimizer_summary = {"present": bool(optimizer)}
    if optimizer:
        state = optimizer[0]
        group = state.get("param_groups", [{}])[0]
        state_tensors = [
            value
            for parameter_state in state.get("state", {}).values()
            for value in parameter_state.values()
            if torch.is_tensor(value)
        ]
        optimizer_summary.update({
            "lr": group.get("lr"),
            "weight_decay": group.get("weight_decay"),
            "step_min": min(
                (
                    int(parameter_state["step"])
                    for parameter_state in state.get("state", {}).values()
                    if "step" in parameter_state
                ),
                default=None,
            ),
            "step_max": max(
                (
                    int(parameter_state["step"])
                    for parameter_state in state.get("state", {}).values()
                    if "step" in parameter_state
                ),
                default=None,
            ),
            "all_state_tensors_finite": all(
                torch.isfinite(value).all() for value in state_tensors
            ),
        })

    schedulers = checkpoint.get("lr_schedulers", [])
    return {
        "lightning_epoch_zero_based": checkpoint.get("epoch"),
        "human_epoch": (
            int(checkpoint["epoch"]) + 1 if "epoch" in checkpoint else None
        ),
        "global_step": checkpoint.get("global_step"),
        "ema_raw_compared_parameters": compared,
        "ema_raw_relative_l2": (
            math.sqrt(squared_delta / max(squared_ema, 1e-30))
        ),
        "optimizer": optimizer_summary,
        "scheduler": schedulers[0] if schedulers else {"present": False},
    }


def _trajectory(path: Path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    numeric = [{
        "epoch": int(row["epoch"]),
        "train_loss": float(row["mean_train_loss"]),
        "val_loss": float(row["val_loss"]),
        "strict50": float(row["val_match_strict50"]),
        "lr": float(row["lr"]),
    } for row in rows]
    fit_rows = numeric[-8:]
    x = np.asarray([1.0 / math.sqrt(row["epoch"]) for row in fit_rows])
    design = np.column_stack([np.ones(len(x)), x])
    forecasts = {}
    for key in ("val_loss", "strict50"):
        values = np.asarray([row[key] for row in fit_rows])
        coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
        forecasts[key] = {
            str(epoch): float(coefficients @ [1.0, 1.0 / math.sqrt(epoch)])
            for epoch in (24, 32, 40)
        }
    return {
        "completed_epochs": len(numeric),
        "latest": numeric[-1],
        "all_finite": all(
            math.isfinite(value)
            for row in numeric for key, value in row.items() if key != "epoch"
        ),
        "val_loss_best_epoch": min(
            numeric, key=lambda row: row["val_loss"]
        )["epoch"],
        "train_val_loss_gap_latest": (
            numeric[-1]["train_loss"] - numeric[-1]["val_loss"]
        ),
        "inverse_sqrt_forecast": forecasts,
        "forecast_fit_epochs": [
            fit_rows[0]["epoch"], fit_rows[-1]["epoch"]
        ],
        "forecast_caveat": (
            "descriptive extrapolation only; scheduler changes and terminal "
            "annealing are not represented"
        ),
    }


def _step_components(path: Path, completed_human_epochs: int):
    frame = pl.read_csv(path, ignore_errors=True)
    component_columns = [
        "train/L_att", "train/L_rep", "train/L_var",
        "train/L_beta_sig", "train/L_beta_noise",
        "train/L_beta_suppress", "train/loss",
    ]
    available = [
        column for column in component_columns if column in frame.columns
    ]
    completed = frame.filter(
        pl.col("epoch").is_not_null()
        & (pl.col("epoch") < completed_human_epochs)
    )
    means = (
        completed
        .group_by("epoch")
        .agg([
            pl.col(column).drop_nulls().mean().alias(column)
            for column in available
        ])
        .sort("epoch")
    )
    nan_skips = (
        float(completed["train/nan_skip"].drop_nulls().sum())
        if "train/nan_skip" in frame.columns else 0.0
    )
    return {
        "component_epoch_means": means.to_dicts(),
        "nan_skip_count_logged": nan_skips,
        "all_component_means_finite": all(
            np.isfinite(value)
            for row in means.to_dicts()
            for key, value in row.items()
            if key != "epoch" and value is not None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epoch-csv", type=Path, required=True)
    parser.add_argument("--step-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("CUDA is visible; set CUDA_VISIBLE_DEVICES empty")
    torch.set_num_threads(2)

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model, _ = _build_model(checkpoint)
    dataset = IDEAParquetDataset(
        args.data_dir, seed_range=(181, 182), max_hits_per_event=None,
        with_time=False, drop_loopers=False, merge_daughters=False,
        with_drift_dir=model.needs_drift_dir,
    )
    indices = []
    for target_size in (500, 1000, 2000):
        indices.append(_choose_event(dataset, target_size, set(indices)))
    events = [(index, dataset[index]) for index in indices]
    gradient_index, gradient_event = min(
        events, key=lambda item: abs(int(item[1]["n_hits"]) - 1000)
    )
    model.eval()
    radius_events = []
    for index, event in events:
        sensitivity = _radius_sensitivity(
            model, event["features"], [event["n_hits"]]
        )
        sensitivity.update({
            "dataset_index": index,
            "event_id": int(dataset._index[index][2]),
            "n_hits": int(event["n_hits"]),
        })
        radius_events.append(sensitivity)
    sensitivity_keys = [
        "embedding_relative_rms_zero_radius",
        "embedding_relative_rms_plus_10_percent",
        "first_layernorm_relative_rms_zero_radius",
        "first_layernorm_relative_rms_plus_10_percent",
        "first_layernorm_repeatability_relative_rms",
        "checkpoint_output_relative_rms_zero_radius",
        "checkpoint_output_relative_rms_plus_10_percent",
        "checkpoint_output_repeatability_relative_rms",
        "coordinate_output_relative_rms_zero_radius",
        "beta_logit_output_relative_rms_zero_radius",
    ]
    trajectory = _trajectory(args.epoch_csv)
    radius_mean = {
        key: float(np.mean([event[key] for event in radius_events]))
        for key in sensitivity_keys
    }
    materiality_threshold = 1e-5
    repeatability_floor = max(
        radius_mean["checkpoint_output_repeatability_relative_rms"],
        1e-12,
    )
    radius_material = (
        radius_mean["checkpoint_output_relative_rms_zero_radius"]
        >= materiality_threshold
        and radius_mean["checkpoint_output_relative_rms_zero_radius"]
        >= 10 * repeatability_floor
    )
    report = {
        "scope": "immutable epoch-16 EMA checkpoint, CPU only",
        "checkpoint": _checkpoint_state(checkpoint, model),
        "event": {
            "dataset_index": gradient_index,
            "event_id": int(dataset._index[gradient_index][2]),
            "n_hits": int(gradient_event["n_hits"]),
        },
        "radius_sensitivity": {
            "events": len(radius_events),
            "per_event": radius_events,
            "mean": {
                **radius_mean,
            },
            "maximum": {
                key: float(np.max([event[key] for event in radius_events]))
                for key in sensitivity_keys
            },
            "normalization_configuration": radius_events[0][
                "normalization_configuration"
            ],
            "materiality_test": {
                "output_relative_rms_threshold": materiality_threshold,
                "minimum_signal_to_repeatability_ratio": 10.0,
                "observed_signal_to_repeatability_ratio": (
                    radius_mean[
                        "checkpoint_output_relative_rms_zero_radius"
                    ] / repeatability_floor
                ),
                "radius_response_material": radius_material,
            },
        },
        "gradient_reachability": _gradient_inventory(
            model, gradient_event
        ),
        "trajectory": trajectory,
        "step_metrics": _step_components(
            args.step_csv, trajectory["completed_epochs"]
        ),
        "instrumentation_limits": {
            "gradient_clipping": (
                "configured at 1.0 by Trainer but clipped-gradient counts are "
                "not logged"
            ),
            "nonfinite_steps": (
                "loss NaN skips are logged only when triggered; optimizer "
                "state finiteness is checked here"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "checkpoint": report["checkpoint"],
        "radius_sensitivity": report["radius_sensitivity"],
        "gradient_reachability": {
            key: value
            for key, value in report["gradient_reachability"].items()
            if key != "per_tensor"
        },
        "trajectory": report["trajectory"],
    }, indent=2))


if __name__ == "__main__":
    main()
