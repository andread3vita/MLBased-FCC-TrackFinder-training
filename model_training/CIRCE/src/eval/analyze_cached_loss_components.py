"""Measure loss values and output-gradient pressure on a forward cache.

This is diagnostic only. It differentiates each current OC component with
respect to the learned clustering coordinates and beta probabilities, without
requiring another model forward pass.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl
import torch

from src.model import object_condensation_loss


def _rms(gradient: torch.Tensor | None) -> float:
    if gradient is None:
        return 0.0
    return float(gradient.square().mean().sqrt())


def _track_separation_weight(
    hits: pl.DataFrame, particles: pl.DataFrame
) -> torch.Tensor:
    target_ids = (
        hits.filter(pl.col("n_hits_total") >= 3)["mc_index"].unique().to_list()
    )
    particles = (
        particles.filter(pl.col("mc_index").is_in(target_ids))
        .sort("mc_index")
    )
    if len(particles) <= 1:
        weight_by_id = {
            int(i): 1.0 / 1.001 for i in particles["mc_index"].to_list()
        }
    else:
        theta = torch.tensor(particles["theta"].to_numpy(), dtype=torch.float32)
        phi = torch.tensor(particles["phi"].to_numpy(), dtype=torch.float32)
        eta_phi = torch.stack([-torch.log(torch.tan(theta / 2)), phi], dim=1)
        nearest = torch.cdist(eta_phi, eta_phi).sort(dim=1).values[:, 1]
        weights = 1.0 / (nearest + 0.001)
        weight_by_id = dict(zip(
            particles["mc_index"].to_list(), weights.tolist()
        ))
    return torch.tensor(
        [weight_by_id.get(int(i), 1.0) for i in hits["mc_index"]],
        dtype=torch.float32,
    )


def _event_diagnostics(
    df: pl.DataFrame, embed_dim: int, oc_mode: str,
    particles: pl.DataFrame | None,
) -> dict:
    coords = torch.tensor(
        df.select([f"coord_{i}" for i in range(embed_dim)]).to_numpy(),
        dtype=torch.float32,
        requires_grad=True,
    )
    beta = torch.tensor(
        df["beta"].to_numpy(), dtype=torch.float32, requires_grad=True
    )
    mc = torch.tensor(df["mc_index"].to_numpy(), dtype=torch.long)
    n_hits_total = torch.tensor(df["n_hits_total"].to_numpy(), dtype=torch.long)
    mc = mc.clone()
    mc[n_hits_total < 3] = -1
    batch = torch.zeros(len(df), dtype=torch.long)
    track_weight = None
    if oc_mode == "ggtf":
        if particles is None:
            raise ValueError("--mc-signal is required when auditing GGTF mode")
        track_weight = _track_separation_weight(df, particles)
    total, components = object_condensation_loss(
        coords,
        beta,
        mc,
        batch,
        noise_index=-1,
        qmin=0.1,
        attr_weight=1.0,
        repul_weight=1.0,
        beta_suppress_weight=0.1,
        var_weight=0.3,
        return_components=True,
        detach_components=False,
        oc_mode=oc_mode,
        track_separation_weight=track_weight,
    )
    objectives = {
        "attraction": components["L_V_att"],
        "repulsion": components["L_V_rep"],
        "beta_signal": components["L_beta_sig"],
        "beta_noise": components["L_beta_noise"],
        "beta_nonalpha_suppression": components["L_beta_suppress"],
        "variance_weighted": 0.3 * components["L_var"],
        "total": total,
    }
    result = {
        "n_hits": len(df),
        "n_targets": int(torch.unique(mc[mc >= 0]).numel()),
    }
    for name, value in objectives.items():
        if value.requires_grad:
            grad_coords, grad_beta = torch.autograd.grad(
                value, (coords, beta), retain_graph=True, allow_unused=True
            )
        else:
            # Some valid events make a component identically zero: for
            # example, beta-noise when every cached hit is signal, or
            # repulsion when only one target survives. A constant has exactly
            # zero output-gradient pressure; autograd cannot differentiate it.
            grad_coords, grad_beta = None, None
        result[name] = {
            "value": float(value.detach()),
            "coord_gradient_rms": _rms(grad_coords),
            "beta_gradient_rms": _rms(grad_beta),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embed-dim", type=int, default=4)
    parser.add_argument("--events", type=int, default=64)
    parser.add_argument("--mc-signal")
    parser.add_argument(
        "--modes", nargs="+", default=["paper_hinge"],
        choices=["paper_hinge", "ggtf"],
    )
    args = parser.parse_args()

    df = pl.read_parquet(args.cache)
    keys = (
        df.select(["seed", "event_id"])
        .unique()
        .sort(["seed", "event_id"])
        .head(args.events)
    )
    mc = pl.read_parquet(args.mc_signal) if args.mc_signal else None
    report = {}
    for mode in args.modes:
        rows = []
        for seed, event_id in keys.iter_rows():
            event = df.filter(
                (pl.col("seed") == seed) & (pl.col("event_id") == event_id)
            )
            particles = None if mc is None else mc.filter(
                (pl.col("seed") == seed) & (pl.col("event_id") == event_id)
            )
            rows.append(_event_diagnostics(
                event, args.embed_dim, mode, particles
            ))

        aggregate = {
            "events": len(rows),
            "mean_hits": float(np.mean([row["n_hits"] for row in rows])),
            "mean_targets": float(np.mean([row["n_targets"] for row in rows])),
            "components": {},
        }
        names = [name for name in rows[0] if isinstance(rows[0][name], dict)]
        for name in names:
            aggregate["components"][name] = {
                key: float(np.mean([row[name][key] for row in rows]))
                for key in ("value", "coord_gradient_rms", "beta_gradient_rms")
            }
        report[mode] = aggregate
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
