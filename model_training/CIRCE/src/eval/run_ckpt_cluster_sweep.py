"""Checkpoint-agnostic CPU-only clustering study over an immutable forward cache.

This generalises the epoch-16 study (run_epoch16_cluster_sweep.py, which stays
untouched so its locked campaign remains reproducible) and closes two coverage
gaps that study had:

  1. The density clusterers only ever received 19 points against 234 for the
     greedy family, and their only real knob (td) was scanned on the coarse
     greedy grid rather than in the fine range where they actually operate.
  2. The merge/attach postprocessing refinement was applied to the greedy
     family only, so dbscan/hdbscan were never combined with it at all.

Here every clusterer gets the same refinement treatment. Stages run in
increasing cost order so the density answer lands before the long greedy grid.

No model forward pass and no holdout data are touched; the cache is read-only
and every point is idempotent, writing its own log and provenance.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Greedy-family grid, identical to the epoch-16 study for direct comparability.
TBETAS = (0.05, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
TDS = (0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225,
       0.25, 0.3, 0.4, 0.6, 0.8)

# Density clusterers ignore tbeta; td is eps (dbscan) or cluster_selection_epsilon
# (hdbscan). At epoch 16 their whole useful range was td <= 0.15, so scan that
# region finely instead of reusing the coarse greedy ladder.
DENSITY_TDS = (0.02, 0.03, 0.04, 0.05, 0.0625, 0.075, 0.0875, 0.1,
               0.125, 0.15, 0.175, 0.2, 0.25, 0.3)
DENSITY_MIN_SAMPLES = (3, 4, 5)
DENSITY_TBETA = 0.5  # unused by the algorithm; fixed so point names stay stable

GREEDY_CLUSTERERS = ("greedy", "self_seed_greedy")
DENSITY_CLUSTERERS = ("dbscan", "hdbscan")

# Source files whose content defines the meaning of every number produced here.
EVALUATOR_FILES = (
    "src/eval/run_ckpt_cluster_sweep.py",
    "src/eval/fcc_cache_parallel.py",
    "src/eval_fcc_metrics_v36.py",
    "src/eval/ggtf_assign.py",
    "src/eval/helix_merge.py",
)


def _slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _source_hashes(package: Path) -> dict[str, str]:
    return {name: _sha256(package / name) for name in EVALUATOR_FILES}


def _point_config(
    clusterer: str, tbeta: float, td: float, min_hits: int,
    merge_td: float, attach_td: float, helix_tol: float,
) -> dict:
    return {
        "clusterer": clusterer,
        "tbeta": float(tbeta),
        "td": float(td),
        "min_cluster_hits": int(min_hits),
        "merge_td": float(merge_td),
        "attach_td": float(attach_td),
        "helix_tol": float(helix_tol),
        "min_target_hits": 3,
        "beta_mode": "sigmoid",
    }


def _complete(output: Path, provenance: str, config: dict) -> bool:
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if summary.get("sweep_provenance_sha256") != provenance:
        return False
    return all(summary.get(key) == value for key, value in config.items())


def _preflight(args) -> dict:
    cache_parquet = args.cache / "forward_hits.parquet"
    manifest_path = args.cache / "manifest.json"
    for path in (cache_parquet, manifest_path, args.mc_signal):
        if not path.is_file():
            raise RuntimeError(f"missing required input: {path}")
    manifest = json.loads(manifest_path.read_text())
    checkpoint_sha = manifest.get("checkpoint_sha256")
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise RuntimeError(
            "cache was produced by a different checkpoint than expected: "
            f"{checkpoint_sha} != {args.expected_checkpoint_sha256}"
        )
    campaign = {
        "label": args.label,
        "source_sha256": _source_hashes(args.package),
        "inputs": {
            "forward_cache": {
                "path": str(cache_parquet),
                "sha256": _sha256(cache_parquet),
            },
            "mc_signal": {
                "path": str(args.mc_signal),
                "sha256": _sha256(args.mc_signal),
            },
        },
        "checkpoint_sha256": checkpoint_sha,
        "cache_manifest": manifest,
        "grid": {
            "tbetas": list(TBETAS),
            "tds": list(TDS),
            "density_tds": list(DENSITY_TDS),
            "density_min_samples": list(DENSITY_MIN_SAMPLES),
        },
    }
    campaign["campaign_sha256"] = _canonical_hash(campaign)
    campaign_path = args.output / "campaign_manifest.json"
    if campaign_path.exists():
        existing = json.loads(campaign_path.read_text())
        if existing != campaign:
            raise RuntimeError(
                "existing campaign manifest does not match current code/inputs"
            )
    else:
        if any(args.output.glob("*/*/*/summary.json")):
            raise RuntimeError(
                "refusing to adopt existing summaries without campaign provenance"
            )
        campaign_path.write_text(json.dumps(campaign, indent=2) + "\n")
    return campaign


def _assert_campaign_current(args) -> None:
    if _source_hashes(args.package) != args.campaign["source_sha256"]:
        raise RuntimeError("evaluator source changed mid-campaign")
    for identity in args.campaign["inputs"].values():
        if _sha256(Path(identity["path"])) != identity["sha256"]:
            raise RuntimeError(f"campaign input changed: {identity['path']}")


def _run_point(
    args, stage: str, clusterer: str, tbeta: float, td: float,
    min_hits: int = 4, merge_td: float = 0.0, attach_td: float = 0.0,
    helix_tol: float = 0.0,
) -> Path:
    name = (
        f"tb{_slug(tbeta)}_td{_slug(td)}_mh{min_hits}"
        f"_m{_slug(merge_td)}_a{_slug(attach_td)}"
    )
    if helix_tol > 0:
        name += f"_h{_slug(helix_tol)}"
    output = args.output / stage / clusterer / name
    output.mkdir(parents=True, exist_ok=True)
    config = _point_config(
        clusterer, tbeta, td, min_hits, merge_td, attach_td, helix_tol
    )
    if _complete(output, args.provenance_sha256, config):
        return output
    _assert_campaign_current(args)
    command = [
        sys.executable, "-u", str(args.package / "src/eval/fcc_cache_parallel.py"),
        "--cache_path", str(args.cache),
        "--mc_signal", str(args.mc_signal),
        "--embed_dim", str(args.embed_dim),
        "--clusterer", clusterer,
        "--tbeta", str(tbeta),
        "--td", str(td),
        "--beta_mode", "sigmoid",
        "--min_cluster_hits", str(min_hits),
        "--min_target_hits", "3",
        "--workers", str(args.workers),
        "--merge_td", str(merge_td),
        "--attach_td", str(attach_td),
        "--helix_tol", str(helix_tol),
        "--helix_min_hits", "10",
        "--summary_only",
        "--output_dir", str(output),
        "--tag", f"{args.label}_{stage}_{clusterer}",
        "--sweep_provenance_sha256", args.provenance_sha256,
    ]
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(args.package),
        "CUDA_VISIBLE_DEVICES": "",
        "FIX_PARTICLE_ZERO": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "POLARS_MAX_THREADS": "4",
    })
    started = time.time()
    with (output / "run.log").open("w") as log:
        result = subprocess.run(
            command, cwd=args.package, env=env, stdout=log,
            stderr=subprocess.STDOUT, check=False,
        )
    if result.returncode:
        try:
            tail = (output / "run.log").read_text(
                errors="replace"
            ).splitlines()[-10:]
        except OSError:
            tail = []
        if tail:
            print("--- failing point log tail ---", file=sys.stderr)
            print("\n".join(tail), file=sys.stderr)
            print("------------------------------", file=sys.stderr)
        raise RuntimeError(
            f"point failed ({clusterer}, tbeta={tbeta}, td={td}, "
            f"mh={min_hits}, m={merge_td}, a={attach_td}); see {output}/run.log"
        )
    if not _complete(output, args.provenance_sha256, config):
        raise RuntimeError(f"point wrote mismatched provenance: {output}")
    print(
        f"[{stage}] {clusterer:<18} tb={tbeta:<5g} td={td:<6g} "
        f"mh={min_hits} m={merge_td:g} a={attach_td:g} "
        f"{time.time() - started:.1f}s",
        flush=True,
    )
    return output


def _metric(summary: dict, *path, default=None):
    node = summary
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _eligible(summary: dict) -> bool:
    """A point is comparable only if the GGTF assignment produced matches."""
    return (
        _metric(summary, "ggtf_n_matched_gt10", default=0) > 0
        and summary.get("promotable_operating_point", True)
    )


def _load_stage_points(root: Path, stage: str) -> list[tuple[Path, dict]]:
    points = []
    for path in sorted((root / stage).glob("*/*/summary.json")):
        points.append((path.parent, json.loads(path.read_text())))
    return points


def _pareto_pair(points, x_getter, y_getter):
    """Non-dominated points maximising x while minimising y."""
    points = [item for item in points if _eligible(item[1])]
    frontier = []
    for candidate in points:
        cx, cy = x_getter(candidate[1]), y_getter(candidate[1])
        if cx is None or cy is None:
            continue
        dominated = False
        for other in points:
            ox, oy = x_getter(other[1]), y_getter(other[1])
            if ox is None or oy is None:
                continue
            if ox >= cx and oy <= cy and (ox > cx or oy < cy):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier


def _refinement_bases(points, cap: int) -> list[tuple[Path, dict]]:
    """Pick non-dominated bases on the efficiency/fake-rate trade-offs."""
    ggtf = lambda s: _metric(s, "ggtf_fake_rate_gt10")
    objectives = (
        (lambda s: _metric(s, "idea", "match_rate"), ggtf),
        (lambda s: _metric(s, "idea", "def2"), ggtf),
        (lambda s: _metric(s, "low_pt_idea", "0.1-0.2", "def2"), ggtf),
    )
    chosen: dict[str, tuple[Path, dict]] = {}
    for x_getter, y_getter in objectives:
        for path, summary in _pareto_pair(points, x_getter, y_getter):
            chosen[str(path)] = (path, summary)
    ranked = sorted(
        chosen.values(),
        key=lambda item: _metric(item[1], "idea", "def2") or 0.0,
        reverse=True,
    )
    return ranked[:cap]


def _refine(args, stage: str, bases, provenance_path: Path) -> None:
    provenance = []
    for base_path, summary in bases:
        clusterer = summary["clusterer"]
        tbeta = summary["tbeta"]
        td = summary["td"]
        base_hits = summary["min_cluster_hits"]
        provenance.append({
            "source": str(base_path),
            "clusterer": clusterer,
            "tbeta": tbeta,
            "td": td,
            "min_cluster_hits": base_hits,
        })
        variants = [
            (max(3, base_hits - 1), 0.0, 0.0),
            (base_hits + 1, 0.0, 0.0),
            (base_hits, 0.5 * td, 0.0),
            (base_hits, td, 0.0),
            (base_hits, 0.0, 0.5 * td),
            (base_hits, 0.0, td),
            (base_hits, 0.5 * td, 0.5 * td),
        ]
        for min_hits, merge, attach in variants:
            _run_point(
                args, stage, clusterer, tbeta, td,
                min_hits=min_hits, merge_td=merge, attach_td=attach,
            )
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--mc-signal", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--embed-dim", type=int, default=4)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--refine-cap", type=int, default=10)
    parser.add_argument(
        "--skip-greedy-grid", action="store_true",
        help="run only the density stages and their refinement",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    lock_handle = (args.output / "campaign.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another sweep process holds the campaign lock") from exc

    args.campaign = _preflight(args)
    args.provenance_sha256 = args.campaign["campaign_sha256"]
    print(f"campaign {args.provenance_sha256[:12]} label={args.label}", flush=True)

    # Stage 1: density scan. Cheapest and the previously under-explored family.
    for clusterer in DENSITY_CLUSTERERS:
        for td in DENSITY_TDS:
            for min_samples in DENSITY_MIN_SAMPLES:
                _run_point(
                    args, "density", clusterer, DENSITY_TBETA, td,
                    min_hits=min_samples,
                )

    # Stage 2: density refinement, including the merge/attach postprocessing the
    # epoch-16 study never applied to these clusterers.
    _refine(
        args, "density_refinement",
        _refinement_bases(_load_stage_points(args.output, "density"), args.refine_cap),
        args.output / "density_refinement_sources.json",
    )

    if not args.skip_greedy_grid:
        # Stage 3: greedy family grid, identical to epoch 16.
        for clusterer in GREEDY_CLUSTERERS:
            for tbeta in TBETAS:
                for td in TDS:
                    _run_point(args, "full_grid", clusterer, tbeta, td)

        # Stage 4: greedy refinement.
        _refine(
            args, "refinement",
            _refinement_bases(
                _load_stage_points(args.output, "full_grid"), args.refine_cap
            ),
            args.output / "refinement_sources.json",
        )

    _assert_campaign_current(args)
    (args.output / "SWEEP_COMPLETE").write_text(
        json.dumps({
            "campaign_sha256": args.provenance_sha256,
            "label": args.label,
            "greedy_grid_included": not args.skip_greedy_grid,
            "completed_at_unix": time.time(),
        }, indent=2) + "\n"
    )
    print("sweep complete", flush=True)


if __name__ == "__main__":
    main()
