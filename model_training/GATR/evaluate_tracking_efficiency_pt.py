#!/usr/bin/env python3
"""Evaluate tracking efficiency versus truth pT and production displacement.

The evaluator supports unified GATr v1.4.2 checkpoints with either geometry-
only or optional detector-scalar inputs. The
embedding dimension is read from ``clustering.weight`` in the checkpoint; it
is never assumed to be three or five.

Current training checkpoints contain both the raw model state and an
exponential-moving-average (EMA) state used during validation.  Evaluation
uses EMA by default when it is available so that offline metrics reproduce
the weights used for validation and checkpoint selection.

Clustering and particle/track matching use ``src.layers.tracking_metrics``.
Those routines implement the beta-ordered clustering and one-to-one Hungarian
matching from ``src/layers/inference_oc_tracks.py``.  In particular,
``double_majority`` requires both hit efficiency and hit purity >= 50%, while
``idea`` requires hit purity >= 75%.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.utils.detector_features import (
    DEFAULT_LAYERS_PER_SUPERLAYER,
    DETECTOR_FEATURE_NAMES,
    validate_layers_per_superlayer,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_GLOB = (
    "/eos/experiment/fcc/ee/simulation/key4hep_2026_07_29/91GeV/"
    "IDEA_o1_v4/Zuds/graph/*.parquet"
)
DEFAULT_CHECKPOINT = "/afs/cern.ch/work/a/adevita/public/trainingFolder/GATr_990files_IDEAv4o1_standardLoss_5dimEmbeddingSpace_parquet/_epoch=3_step=100000.ckpt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot GATr tracking efficiency versus truth-particle pT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--weights-source",
        "--weights",
        dest="weights",
        choices=("ema", "raw"),
        default="ema",
        help=(
            "checkpoint weights to evaluate: 'ema' requires EMA weights and "
            "'raw' uses state_dict"
        ),
    )
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    parser.add_argument(
        "--data-config",
        default=None,
        help=(
            "data YAML; by default it is selected from the checkpoint's "
            "geometry-only or detector-feature architecture"
        ),
    )
    parser.add_argument(
        "--layers-per-superlayer",
        type=int,
        nargs=14,
        default=None,
        metavar="N",
        help=(
            "override the checkpoint's 14-entry superlayer layout; new "
            "checkpoints store the training layout"
        ),
    )
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "tracking_efficiency_eval"))
    parser.add_argument("--output-stem", default="tracking_efficiency_vs_pt")
    parser.add_argument("--max-files", type=int, default=2)
    parser.add_argument(
        "--max-events",
        type=int,
        default=-1,
        help="global event cap across the selected files; negative means all events",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device, e.g. cuda, cuda:1, or cpu",
    )
    parser.add_argument(
        "--tbeta", type=float, default=0.35,
        help="minimum sigmoid(beta) for a condensation-point seed",
    )
    parser.add_argument(
        "--td", type=float, default=0.2,
        help="maximum embedding-space distance for assigning a hit",
    )
    parser.add_argument("--min-hits", type=int, default=3, help="minimum reconstructed-track size")
    parser.add_argument(
        "--rejected-seed-policy",
        choices=("discard", "keep", "attach-after-accept"),
        default="discard",
        help="handling of a seed whose candidate is smaller than min-hits",
    )
    parser.add_argument(
        "--truth-min-hits",
        type=int,
        default=3,
        help="minimum number of retained detector hits for a truth particle",
    )
    parser.add_argument(
        "--matching-metric",
        choices=("double_majority", "idea"),
        default="double_majority",
    )
    parser.add_argument("--pt-min", type=float, default=0.1)
    parser.add_argument("--pt-max", type=float, default=60.0)
    parser.add_argument("--pt-log-step", type=float, default=0.2)
    parser.add_argument("--displacement-min", type=float, default=0.0)
    parser.add_argument("--displacement-max", type=float, default=2000.0)
    parser.add_argument("--displacement-bin-width", type=float, default=50.0)
    parser.add_argument(
        "--displacement-output-stem",
        default="tracking_efficiency_vs_displacement",
    )
    parser.add_argument("--theta-min", type=float, default=10.0, help="degrees, exclusive")
    parser.add_argument("--theta-max", type=float, default=170.0, help="degrees, exclusive")
    parser.add_argument(
        "--gen-status",
        type=int,
        nargs="+",
        default=(0, 1),
        help="accepted generator-status values",
    )
    parser.add_argument(
        "--position-scale",
        type=float,
        default=None,
        help="override the modern model position scale (otherwise use checkpoint metadata or 1000)",
    )
    return parser


def _load_checkpoint(
    path: str, weights: str = "auto"
) -> Tuple[dict, Dict[str, torch.Tensor], dict, str]:
    if weights not in {"auto", "ema", "raw"}:
        raise ValueError(f"Unknown checkpoint weight selection: {weights!r}")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {path} is not a dictionary")

    ema_state = checkpoint.get("ema_state_dict")
    has_ema = isinstance(ema_state, Mapping) and bool(ema_state)
    if weights == "ema" and not has_ema:
        raise ValueError(
            f"Checkpoint {path} does not contain a non-empty ema_state_dict"
        )

    if weights == "ema" or (weights == "auto" and has_ema):
        selected_state = ema_state
        weight_source = "ema"
    else:
        selected_state = checkpoint.get("state_dict", checkpoint)
        weight_source = "raw"
    if not isinstance(selected_state, Mapping):
        raise TypeError(f"Checkpoint {path} has no tensor {weight_source} state dict")

    clustering_keys = [
        key for key in selected_state if key.endswith("clustering.weight")
    ]
    if len(clustering_keys) != 1:
        raise ValueError(
            "Expected exactly one key ending in 'clustering.weight', found "
            f"{clustering_keys}"
        )
    prefix = clustering_keys[0][: -len("clustering.weight")]
    state = {
        (key[len(prefix) :] if prefix and key.startswith(prefix) else key): value
        for key, value in selected_state.items()
        if torch.is_tensor(value)
    }
    hparams = checkpoint.get("hyper_parameters") or {}
    return (
        checkpoint,
        state,
        dict(hparams) if isinstance(hparams, Mapping) else {},
        weight_source,
    )


def _architecture_from_state(state: Mapping[str, torch.Tensor]) -> dict:
    clustering = state["clustering.weight"]
    if clustering.ndim != 2:
        raise ValueError("clustering.weight must be a matrix")
    block_indices = {
        int(match.group(1))
        for key in state
        if (match := re.match(r"gatr\.blocks\.(\d+)\.", key))
    }
    if not block_indices:
        raise ValueError("No gatr.blocks.* parameters found in checkpoint")

    if "ScaledGooeyBatchNorm2_1.weight" in state:
        raise ValueError(
            "Legacy pre-v1.4.2 checkpoints are no longer supported by the "
            "unified GATr evaluator"
        )

    hidden_s_weight = state.get("gatr.linear_in.mvs2s.weight")
    if hidden_s_weight is None:
        raise ValueError("Cannot infer hidden scalar channels from checkpoint")
    return {
        "embedding_dim": int(clustering.shape[0]),
        "hidden_mv_channels": int(clustering.shape[1]),
        "hidden_s_channels": int(hidden_s_weight.shape[0]),
        "num_blocks": max(block_indices) + 1,
        "detector_scalar_dim": int(
            state["gatr.linear_in.s2mvs.weight"].shape[1]
            if "gatr.linear_in.s2mvs.weight" in state
            else 0
        ),
        "has_helix_head": "helix_proxy.weight" in state,
    }


class CheckpointGATr(nn.Module):
    """Small inference-only wrapper reconstructed entirely from checkpoint shapes."""

    def __init__(
        self,
        architecture: Mapping[str, object],
        device: torch.device,
        position_scale: float,
    ) -> None:
        super().__init__()
        # Imports are kept here so --help and checkpoint inspection remain useful
        # even in an environment missing optional GATr runtime packages.
        gatr_root = REPO_ROOT / "src" / "gatr_v142"
        if str(gatr_root) not in sys.path:
            sys.path.insert(0, str(gatr_root))
        from gatr.interface import embed_point, embed_scalar, embed_translation
        from gatr.layers.attention.config import SelfAttentionConfig
        from gatr.layers.mlp.config import MLPConfig
        from gatr.nets.gatr import GATr

        self._embed_point = embed_point
        self._embed_scalar = embed_scalar
        self._embed_translation = embed_translation
        self.embedding_dim = int(architecture["embedding_dim"])
        self.detector_scalar_dim = int(architecture["detector_scalar_dim"])
        self.position_scale = float(position_scale)

        hidden_mv = int(architecture["hidden_mv_channels"])
        self.gatr = GATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv,
            in_s_channels=(self.detector_scalar_dim or None),
            out_s_channels=None,
            hidden_s_channels=int(architecture["hidden_s_channels"]),
            num_blocks=int(architecture["num_blocks"]),
            attention=SelfAttentionConfig(),
            mlp=MLPConfig(),
        )
        self.clustering = nn.Linear(hidden_mv, self.embedding_dim, bias=False)
        self.beta = nn.Linear(hidden_mv, 1)
        self.helix_proxy = (
            nn.Linear(hidden_mv, 4) if bool(architecture["has_helix_head"]) else None
        )

    def forward(self, graph) -> torch.Tensor:
        dtype = self.clustering.weight.dtype
        position = graph.ndata["pos_hits_xyz"].to(dtype=dtype)
        vector = graph.ndata["vector"].to(dtype=dtype)
        hit_type = graph.ndata["hit_type"].reshape(-1, 1).to(dtype=dtype)
        position = position / self.position_scale
        vector = vector / self.position_scale

        multivectors = (
            self._embed_point(position)
            + self._embed_scalar(hit_type)
            + self._embed_translation(vector)
        ).unsqueeze(-2)
        if not self.detector_scalar_dim:
            scalars = None
        else:
            if "scalar_features" not in graph.ndata:
                raise KeyError(
                    "Detector-feature checkpoint requires "
                    "graph.ndata['scalar_features']"
                )
            scalars = graph.ndata["scalar_features"].to(dtype=dtype)
        join_reference = multivectors.mean(dim=0, keepdim=True).expand_as(
            multivectors
        )
        embedded, _ = self.gatr(
            multivectors,
            scalars=scalars,
            attention_mask=None,  # Each inference graph is exactly one event.
            join_reference=join_reference,
        )
        latent = embedded[:, 0, :]
        pieces = [self.clustering(latent), self.beta(latent)]
        if self.helix_proxy is not None:
            pieces.append(self.helix_proxy(latent))
        return torch.cat(pieces, dim=1)


def _natural_key(path: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path)]


def _select_files(pattern: str, max_files: int) -> list[str]:
    files = sorted(glob.glob(pattern), key=_natural_key)
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No input files match {pattern!r}")
    return files


def _event_fraction(files: Iterable[str], max_events: int) -> Tuple[float, int]:
    _ensure_pyarrow()
    try:
        import awkward as ak
        totals = [sum(ak.metadata_from_parquet(path)["col_counts"]) for path in files]
    except ImportError as error:
        raise ImportError(
            "Reading the EOS Parquet input requires pyarrow (used by awkward). "
            "Install pyarrow in the active environment."
        ) from error
    available = int(sum(totals))
    if available == 0:
        raise ValueError("The selected Parquet files contain no events")
    if max_events <= 0:
        return 1.0, available
    return min(1.0, max_events / available), available


def _ensure_pyarrow() -> None:
    """Find pyarrow in a sibling Conda env when this minimal env omits it.

    CERN analysis installations commonly keep the GPU/CPU PyTorch environment
    lean while a neighboring data-processing environment owns pyarrow. This
    local fallback makes the script directly runnable in that setup without
    modifying either environment. A normal installed pyarrow always wins.
    """
    try:
        import pyarrow  # noqa: F401
        return
    except ImportError:
        pass

    python_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    prefix = Path(sys.prefix).resolve()
    envs_root = prefix.parent if prefix.parent.name == "envs" else None
    candidates = (
        sorted(envs_root.glob(f"*/lib/{python_dir}/site-packages/pyarrow"))
        if envs_root is not None
        else []
    )
    for package_dir in candidates:
        site_packages = str(package_dir.parent)
        if site_packages not in sys.path:
            sys.path.append(site_packages)
        importlib.invalidate_caches()
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            continue
        print(f"Using pyarrow from sibling environment: {site_packages}")
        return
    raise ImportError(
        "Reading the EOS Parquet input requires pyarrow. Install it in the "
        "active environment (for example, conda install -c conda-forge pyarrow)."
    )


def _make_dataset(
    files: list[str], config: str, fraction: float, max_events: int,
    layers_per_superlayer: tuple[int, ...],
):
    from src.dataset.dataset import SimpleIterDataset

    return SimpleIterDataset(
        {"evaluation": files},
        config,
        for_training=False,
        load_range_and_fraction=((0.0, 1.0), fraction),
        fetch_by_files=True,
        fetch_step=1,
        async_load=False,
        infinity_mode=False,
        in_memory=False,
        name="tracking_efficiency_evaluation",
        seed=42,
        dataset_cap=max_events if max_events > 0 else None,
        layers_per_superlayer=layers_per_superlayer,
    )


def _particle_info(graph, particle_rows: np.ndarray) -> Dict[int, dict]:
    truth = graph.ndata["particle_number"].detach().cpu().numpy().reshape(-1)
    original = graph.ndata["particle_number_nomap"].detach().cpu().numpy().reshape(-1)
    rows_by_original_id = {
        int(round(float(row[4]))): {
            "theta": float(row[0]),
            "pt": float(row[6]),
            "gen_status": float(row[7]),
            "displacement": float(np.hypot(row[9], row[10])),
        }
        for row in particle_rows
    }
    result: Dict[int, dict] = {}
    for truth_id in np.unique(truth[truth > 0]):
        original_ids = np.unique(original[(truth == truth_id) & (original >= 0)])
        if original_ids.size == 1:
            original_id = int(round(float(original_ids[0])))
            if original_id in rows_by_original_id:
                result[int(truth_id)] = rows_by_original_id[original_id]
    return result


@torch.inference_mode()
def _run_inference(model, dataset, device: torch.device, max_events: int) -> list[dict]:
    events = []
    iterator = iter(dataset)
    while max_events <= 0 or len(events) < max_events:
        try:
            graph, particle_rows = next(iterator)
        except StopIteration:
            break
        graph = graph.to(device)
        output = model(graph)
        expected = model.embedding_dim + 1
        if output.ndim != 2 or output.shape[1] < expected:
            raise ValueError(
                f"Model returned shape {tuple(output.shape)}; expected at least {expected} columns"
            )
        coords = output[:, : model.embedding_dim].float().cpu().numpy()
        beta = torch.sigmoid(output[:, model.embedding_dim]).float().cpu().numpy()
        truth = graph.ndata["particle_number"].detach().cpu().numpy().reshape(-1)
        events.append(
            {
                "coords": coords,
                "beta": beta,
                "truth": truth,
                "particle_info": _particle_info(graph, particle_rows.detach().cpu().numpy()),
            }
        )
        print(
            f"\rProcessed {len(events)} event(s), latest event has {graph.num_nodes()} hits",
            end="",
            flush=True,
        )
    print()
    if not events:
        raise RuntimeError("No valid events were produced by the dataset")
    return events


def main() -> int:
    args = build_parser().parse_args()
    if args.max_files == 0 or args.max_events == 0:
        raise ValueError("--max-files and --max-events must be positive, or negative for no limit")
    if args.pt_min <= 0 or args.pt_max <= args.pt_min or args.pt_log_step <= 0:
        raise ValueError("Require 0 < --pt-min < --pt-max and --pt-log-step > 0")
    if (
        args.displacement_min < 0
        or args.displacement_max <= args.displacement_min
        or args.displacement_bin_width <= 0
    ):
        raise ValueError(
            "Require 0 <= --displacement-min < --displacement-max and "
            "--displacement-bin-width > 0"
        )

    checkpoint, state, hparams, weight_source = _load_checkpoint(
        args.checkpoint, args.weights
    )
    architecture = _architecture_from_state(state)
    if architecture["detector_scalar_dim"] not in (0, len(DETECTOR_FEATURE_NAMES)):
        raise ValueError(
            "Detector checkpoint expects "
            f"{architecture['detector_scalar_dim']} scalar inputs, but the current "
            f"global-layer model expects {len(DETECTOR_FEATURE_NAMES)}."
        )
    saved_layout = hparams.get(
        "layers_per_superlayer", DEFAULT_LAYERS_PER_SUPERLAYER
    )
    layers_per_superlayer = validate_layers_per_superlayer(
        args.layers_per_superlayer
        if args.layers_per_superlayer is not None
        else saved_layout
    )
    position_scale = (
        args.position_scale
        if args.position_scale is not None
        else float(hparams.get("position_scale", 1000.0))
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    print(
        f"Checkpoint weights: {weight_source}\n"
        "Checkpoint architecture: "
        f"embedding_dim={architecture['embedding_dim']}, "
        f"blocks={architecture['num_blocks']}, "
        f"detector_scalar_dim={architecture['detector_scalar_dim']}, "
        f"auxiliary_helix_head={architecture['has_helix_head']}\n"
        f"Layers per superlayer: {layers_per_superlayer}"
    )
    model = CheckpointGATr(
        architecture,
        device=device,
        position_scale=position_scale,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Auxiliary EMA/bookkeeping tensors may exist in full Lightning checkpoints,
    # but every tensor belonging to the reconstructed inference model must load.
    missing_required = [key for key in missing if not key.startswith("helix_proxy.")]
    unexpected_model = [
        key for key in unexpected
        if not key.startswith(("_ema.", "ema.", "criterion."))
    ]
    if missing_required or unexpected_model:
        raise RuntimeError(
            "Checkpoint/model mismatch: "
            f"missing={missing_required}, unexpected={unexpected_model}"
        )
    model.to(device).eval()

    files = _select_files(args.input_glob, args.max_files)
    fraction, available = _event_fraction(files, args.max_events)
    print(
        f"Selected {len(files)} file(s), {available} available events; "
        f"reading the first {fraction:.3%} from each file"
    )
    data_config = args.data_config
    if data_config is None:
        config_name = (
            "config_tracking_parquet_detector.yaml"
            if architecture["detector_scalar_dim"]
            else "config_tracking_parquet.yaml"
        )
        data_config = str(REPO_ROOT / "config_files" / config_name)
    dataset = _make_dataset(
        files,
        data_config,
        fraction,
        args.max_events,
        layers_per_superlayer,
    )
    events = _run_inference(model, dataset, device, args.max_events)

    from src.layers.tracking_metrics import (
        save_tracking_efficiency_displacement_plot,
        save_tracking_efficiency_pt_plot,
        tracking_efficiency_binned_counts,
    )

    pt_bins = np.exp(
        np.arange(np.log(args.pt_min), np.log(args.pt_max), args.pt_log_step)
    )
    if pt_bins[-1] < args.pt_max:
        pt_bins = np.append(pt_bins, args.pt_max)
    displacement_bins = np.arange(
        args.displacement_min,
        args.displacement_max,
        args.displacement_bin_width,
        dtype=np.float64,
    )
    if displacement_bins.size == 0 or displacement_bins[0] != args.displacement_min:
        displacement_bins = np.insert(displacement_bins, 0, args.displacement_min)
    if displacement_bins[-1] < args.displacement_max:
        displacement_bins = np.append(displacement_bins, args.displacement_max)
    counts, missing_info = tracking_efficiency_binned_counts(
        events,
        args.tbeta,
        args.td,
        args.min_hits,
        observables={"pt": pt_bins, "displacement": displacement_bins},
        metric=args.matching_metric,
        truth_min_hits=args.truth_min_hits,
        min_theta=args.theta_min,
        max_theta=args.theta_max,
        gen_status=args.gen_status,
        rejected_seed_policy=args.rejected_seed_policy,
    )
    pt_total, pt_matched = counts["pt"]
    displacement_total, displacement_matched = counts["displacement"]
    output_dir = Path(args.output_dir)
    # Batch jobs often have a read-only home directory. Keep Matplotlib and
    # fontconfig caches beside the requested output instead.
    cache_dir = output_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    image_path = save_tracking_efficiency_pt_plot(
        [
            {
                "name": "gatr",
                "label": (
                    f"GATr ({architecture['embedding_dim']}D, "
                    f"$t_\\beta$={args.tbeta:g}, $t_d$={args.td:g})"
                ),
                "total": pt_total,
                "matched": pt_matched,
            }
        ],
        str(output_dir),
        filename_stem=args.output_stem,
        bins=pt_bins,
        min_x=args.pt_min,
        max_x=args.pt_max,
        min_theta=args.theta_min,
        max_theta=args.theta_max,
        gen_status=args.gen_status,
    )
    displacement_image_path = save_tracking_efficiency_displacement_plot(
        [
            {
                "name": "gatr",
                "label": (
                    f"GATr ({architecture['embedding_dim']}D, "
                    f"$t_\\beta$={args.tbeta:g}, $t_d$={args.td:g})"
                ),
                "total": displacement_total,
                "matched": displacement_matched,
            }
        ],
        str(output_dir),
        filename_stem=args.displacement_output_stem,
        bins=displacement_bins,
        min_x=args.displacement_min,
        max_x=args.displacement_max,
        min_theta=args.theta_min,
        max_theta=args.theta_max,
        gen_status=args.gen_status,
    )
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "weight_source": weight_source,
        "input_files": files,
        "events": len(events),
        "embedding_dim": architecture["embedding_dim"],
        "detector_scalar_dim": architecture["detector_scalar_dim"],
        "detector_feature_names": list(DETECTOR_FEATURE_NAMES),
        "layers_per_superlayer": list(layers_per_superlayer),
        "tbeta": args.tbeta,
        "td": args.td,
        "min_hits": args.min_hits,
        "rejected_seed_policy": args.rejected_seed_policy,
        "truth_min_hits": args.truth_min_hits,
        "matching_metric": args.matching_metric,
        "truth_particles_in_pt_range": int(pt_total.sum()),
        "matched_truth_particles_in_pt_range": int(pt_matched.sum()),
        "integrated_efficiency_in_pt_range": (
            float(pt_matched.sum() / pt_total.sum()) if pt_total.sum() else None
        ),
        "truth_particles_in_displacement_range": int(displacement_total.sum()),
        "matched_truth_particles_in_displacement_range": int(
            displacement_matched.sum()
        ),
        "integrated_efficiency_in_displacement_range": (
            float(displacement_matched.sum() / displacement_total.sum())
            if displacement_total.sum()
            else None
        ),
        "truth_particles_missing_kinematics": int(missing_info["pt"]),
        "truth_particles_missing_displacement": int(missing_info["displacement"]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.output_stem}_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Wrote plot: {image_path}")
    print(f"Wrote binned counts: {output_dir / (args.output_stem + '.csv')}")
    print(f"Wrote displacement plot: {displacement_image_path}")
    print(
        "Wrote displacement counts: "
        f"{output_dir / (args.displacement_output_stem + '.csv')}"
    )
    print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
