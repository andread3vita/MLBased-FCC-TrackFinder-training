"""Audit epoch-16 beta multiplicity, geometry, and loss-output gradients."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.model import object_condensation_loss


PT_EDGES = (0.0, 0.1, 0.2, 0.5, 1.0, 2.0, float("inf"))
BETA_EDGES = (0.0, 0.1, 0.3, 0.5, 0.8, 0.95, 1.01)


def _bin(value, edges):
    for low, high in zip(edges[:-1], edges[1:]):
        if low <= value < high:
            upper = "inf" if not np.isfinite(high) else f"{high:g}"
            return f"{low:g}-{upper}"
    return "outside"


def _track_geometry(frame: pl.DataFrame) -> list[dict]:
    frame = frame.filter(pl.col("n_hits_total") >= 3)
    coords = frame.select([f"coord_{index}" for index in range(4)]).to_numpy()
    beta = frame["beta"].to_numpy()
    truth = frame["mc_index"].to_numpy()
    pt = frame["pt"].to_numpy()
    target_ids = np.unique(truth)
    alpha_indices = np.asarray([
        indices[np.argmax(beta[indices])]
        for target in target_ids
        for indices in [np.flatnonzero(truth == target)]
    ])
    alpha_coords = coords[alpha_indices]
    if len(alpha_coords) > 1:
        alpha_distances = np.linalg.norm(
            alpha_coords[:, None, :] - alpha_coords[None, :, :], axis=-1
        )
        np.fill_diagonal(alpha_distances, np.inf)
        nearest_alpha = alpha_distances.min(axis=1)
    else:
        nearest_alpha = np.full(len(alpha_coords), np.nan)

    rows = []
    for position, (target, alpha_index) in enumerate(
        zip(target_ids, alpha_indices)
    ):
        own = truth == target
        other = ~own
        distances = np.linalg.norm(coords[own] - coords[alpha_index], axis=1)
        other_activity = (
            float(
                np.mean(
                    np.linalg.norm(
                        coords[other] - coords[alpha_index], axis=1
                    ) < 1.0
                )
            )
            if other.any() else 0.0
        )
        rows.append({
            "mc_index": int(target),
            "pt": float(pt[alpha_index]),
            "pt_bin": _bin(float(pt[alpha_index]), PT_EDGES),
            "n_hits": int(own.sum()),
            "alpha_beta": float(beta[alpha_index]),
            "beta_gt_05": int((beta[own] > 0.5).sum()),
            "beta_gt_08": int((beta[own] > 0.8).sum()),
            "beta_gt_095": int((beta[own] > 0.95).sum()),
            "own_compactness_median": float(np.median(distances)),
            "own_compactness_p90": float(np.quantile(distances, 0.9)),
            "nearest_other_alpha": float(nearest_alpha[position]),
            "separation_over_compactness": (
                float(nearest_alpha[position] / max(np.median(distances), 1e-12))
            ),
            "hinge_repulsion_active_other_hit_fraction": other_activity,
        })
    return rows


def _gradient_rows(frame: pl.DataFrame) -> list[dict]:
    coords = torch.tensor(
        frame.select([f"coord_{index}" for index in range(4)]).to_numpy(),
        dtype=torch.float32, requires_grad=True,
    )
    beta = torch.tensor(
        frame["beta"].to_numpy(), dtype=torch.float32, requires_grad=True
    )
    mc = torch.tensor(frame["mc_index"].to_numpy(), dtype=torch.long)
    n_hits_total = torch.tensor(
        frame["n_hits_total"].to_numpy(), dtype=torch.long
    )
    mc[n_hits_total < 3] = -1
    batch = torch.zeros(len(frame), dtype=torch.long)
    total, components = object_condensation_loss(
        coords, beta, mc, batch, noise_index=-1, qmin=0.1,
        attr_weight=1.0, repul_weight=1.0, beta_suppress_weight=0.1,
        var_weight=0.3, return_components=True, detach_components=False,
        oc_mode="paper_hinge",
    )
    objectives = {
        "attraction": components["L_V_att"],
        "repulsion": components["L_V_rep"],
        "beta_signal": components["L_beta_sig"],
        # L_beta_suppress already includes beta_suppress_weight internally.
        "beta_nonalpha_suppression": components["L_beta_suppress"],
        "variance_weighted": 0.3 * components["L_var"],
        "total": total,
    }
    gradients = {}
    for name, value in objectives.items():
        grad_coords, grad_beta = torch.autograd.grad(
            value, (coords, beta), retain_graph=True, allow_unused=True
        )
        beta_probability_gradient = (
            np.zeros(len(frame)) if grad_beta is None
            else grad_beta.detach().numpy()
        )
        beta_values = beta.detach().numpy()
        beta_logit_gradient = (
            beta_probability_gradient * beta_values * (1.0 - beta_values)
        )
        gradients[name] = (
            np.zeros((len(frame), 4)) if grad_coords is None
            else grad_coords.detach().numpy(),
            beta_probability_gradient,
            beta_logit_gradient,
        )

    truth = mc.numpy()
    values = frame["beta"].to_numpy()
    pt = frame["pt"].to_numpy()
    alpha_mask = np.zeros(len(frame), dtype=bool)
    for target in np.unique(truth[truth >= 0]):
        indices = np.flatnonzero(truth == target)
        alpha_mask[indices[np.argmax(values[indices])]] = True
    rows = []
    for index in range(len(frame)):
        row = {
            "pt_bin": _bin(float(pt[index]), PT_EDGES),
            "beta_bin": _bin(float(values[index]), BETA_EDGES),
            "role": (
                "noise_stub" if truth[index] < 0
                else "alpha" if alpha_mask[index]
                else "nonalpha"
            ),
            "hits": 1,
            "beta": float(values[index]),
        }
        for name, (
            coord_gradient,
            beta_probability_gradient,
            beta_logit_gradient,
        ) in gradients.items():
            row[f"{name}_coord_gradient_norm"] = float(
                np.linalg.norm(coord_gradient[index])
            )
            row[f"{name}_beta_probability_gradient_abs"] = float(
                abs(beta_probability_gradient[index])
            )
            row[f"{name}_beta_logit_gradient_abs"] = float(
                abs(beta_logit_gradient[index])
            )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--mc-signal", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gradient-events", type=int, default=32)
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("CUDA is visible; set CUDA_VISIBLE_DEVICES empty")
    torch.set_num_threads(2)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    hits = pl.read_parquet(args.cache)
    particles = pl.read_parquet(args.mc_signal).select(
        ["seed", "event_id", "mc_index", "pt"]
    )
    hits = hits.join(
        particles, on=["seed", "event_id", "mc_index"], how="left",
        validate="m:1",
    ).filter(pl.col("pt").is_not_null())
    keys = hits.select(["seed", "event_id"]).unique().sort(
        ["seed", "event_id"]
    )

    track_rows = []
    gradient_rows = []
    for event_number, (seed, event_id) in enumerate(keys.iter_rows()):
        frame = hits.filter(
            (pl.col("seed") == seed) & (pl.col("event_id") == event_id)
        )
        for row in _track_geometry(frame):
            row["seed"] = seed
            row["event_id"] = event_id
            track_rows.append(row)
        if event_number < args.gradient_events:
            gradient_rows.extend(_gradient_rows(frame))

    track_frame = pl.DataFrame(track_rows)
    track_frame.write_csv(output / "track_diagnostics.csv")
    gradient_frame = pl.DataFrame(gradient_rows)
    gradient_columns = [
        column for column in gradient_frame.columns
        if column.endswith("_gradient_norm")
        or column.endswith("_gradient_abs")
    ]
    gradient_summary = (
        gradient_frame.group_by(["pt_bin", "beta_bin", "role"])
        .agg(
            pl.len().alias("hits"),
            pl.col("beta").mean().alias("mean_beta"),
            *[
                pl.col(column).mean().alias(f"mean_{column}")
                for column in gradient_columns
            ],
        )
        .sort(["pt_bin", "beta_bin", "role"])
    )
    gradient_summary.write_csv(output / "gradient_pressure_by_pt_beta.csv")

    track_summary = (
        track_frame.group_by("pt_bin")
        .agg(
            pl.len().alias("tracks"),
            pl.col("alpha_beta").mean().alias("mean_alpha_beta"),
            pl.col("beta_gt_05").mean().alias("mean_beta_gt_05"),
            pl.col("beta_gt_08").mean().alias("mean_beta_gt_08"),
            pl.col("beta_gt_095").mean().alias("mean_beta_gt_095"),
            pl.col("own_compactness_median").median(),
            pl.col("nearest_other_alpha").median(),
            pl.col("separation_over_compactness").median(),
            pl.col("hinge_repulsion_active_other_hit_fraction").mean(),
        )
        .sort("pt_bin")
    )
    track_summary.write_csv(output / "track_summary_by_pt.csv")

    multiplicity = track_frame["beta_gt_08"].to_numpy()
    report = {
        "scope": (
            "all 500 epoch-16 cached signal-hit events; gradients on first "
            "events are a loss-output surrogate, not a backbone backward pass"
        ),
        "gradient_space": {
            "coordinates": "dL / d(output clustering coordinate)",
            "beta_probability": "dL / d(sigmoid beta)",
            "beta_logit": (
                "dL / d(beta logit) = dL/d(beta) * beta * (1-beta)"
            ),
        },
        "events": len(keys),
        "tracks": len(track_frame),
        "gradient_events": min(args.gradient_events, len(keys)),
        "high_beta_multiplicity": {
            "mean_beta_gt_08_per_track": float(multiplicity.mean()),
            "fraction_tracks_multiple_beta_gt_08": float(
                np.mean(multiplicity > 1)
            ),
            "maximum_beta_gt_08_on_one_track": int(multiplicity.max()),
        },
        "embedding_geometry": {
            "median_own_compactness": float(
                track_frame["own_compactness_median"].median()
            ),
            "median_nearest_other_alpha": float(
                track_frame["nearest_other_alpha"].median()
            ),
            "median_separation_over_compactness": float(
                track_frame["separation_over_compactness"].median()
            ),
            "mean_hinge_repulsion_active_other_hit_fraction": float(
                track_frame[
                    "hinge_repulsion_active_other_hit_fraction"
                ].mean()
            ),
        },
        "artifacts": {
            "track_diagnostics": str(
                (output / "track_diagnostics.csv").resolve()
            ),
            "track_summary_by_pt": str(
                (output / "track_summary_by_pt.csv").resolve()
            ),
            "gradient_pressure": str(
                (output / "gradient_pressure_by_pt_beta.csv").resolve()
            ),
        },
    }
    (output / "embedding_loss_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
