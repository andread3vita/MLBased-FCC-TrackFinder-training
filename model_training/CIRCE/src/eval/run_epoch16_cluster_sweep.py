"""Run the leakage-safe CPU-only epoch-16 clustering study.

The immutable forward cache is clustered repeatedly; no model forward pass and
no holdout data are touched. Every point is idempotent and writes its own log.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import polars as pl


TBETAS = (0.05, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
TDS = (0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225,
       0.25, 0.3, 0.4, 0.6, 0.8)
EVALUATOR_FILES = (
    "src/eval/run_epoch16_cluster_sweep.py",
    "src/eval/run_epoch16_oracle_appendix.py",
    "src/eval/report_epoch16_cluster_sweep.py",
    "src/eval/fcc_cache_parallel.py",
    "src/eval_fcc_metrics_v36.py",
    "src/eval/ggtf_assign.py",
    "src/eval/helix_merge.py",
    "src/eval_sweep_v33.py",
    "src/lowpt_op_sweep.py",
)
EXPECTED_CACHE_FIELDS = {
    "seeds": "181-181",
    "embed_dim": 4,
    "max_hits": 0,
    "min_signal_mc": 0,
    "fix_particle_zero": True,
    "drop_loopers": False,
    "merge_daughters": False,
    "algebra": "conformal",
    "cga_hit_encoding": "sphere_circle",
    "physical_drift_geometry": True,
    "normalize_mv_inputs": False,
    "fix_cga_null": True,
    "fix_wire_dir": True,
    "no_legacy_equivariance": True,
    "equivariance_group": "e3",
    "invariant_output_head": True,
    "n_events": 500,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _canonical_hash(data: dict) -> str:
    encoded = json.dumps(
        data, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_hashes(package: Path) -> dict[str, str]:
    return {
        relative: _sha256(package / relative)
        for relative in EVALUATOR_FILES
    }


def _point_config(
    clusterer: str,
    tbeta: float,
    td: float,
    min_hits: int,
    merge_td: float,
    attach_td: float,
    helix_tol: float,
) -> dict:
    return {
        "clusterer": clusterer,
        "tbeta": tbeta,
        "td": td,
        "min_cluster_hits": min_hits,
        "merge_td": merge_td,
        "attach_td": attach_td,
        "helix_tol": helix_tol,
        "min_target_hits": 3,
        "min_signal_mc": 0,
    }


def _complete(
    path: Path,
    provenance_sha256: str,
    expected_config: dict,
) -> bool:
    summary = path / "summary.json"
    if not summary.exists():
        return False
    try:
        data = json.loads(summary.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (
        data.get("fix_particle_zero") is True
        and data.get("sweep_provenance_sha256") == provenance_sha256
        and "event_splits" in data
        and "ggtf_candidate_fake_fraction_gt10" in data
        and "all_targets_no_reconstruction_cuts" in data
        and all(data.get(key) == value for key, value in expected_config.items())
    )


def _assert_campaign_current(args) -> None:
    if _source_hashes(args.package) != args.campaign["source_sha256"]:
        raise RuntimeError(
            "evaluator source changed after campaign preflight; refusing to mix code"
        )
    for label, expected in args.campaign["inputs"].items():
        path = Path(expected["path"])
        stat = path.stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise RuntimeError(
                f"{label} changed after campaign preflight; refusing to continue"
            )


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
        "--embed_dim", "4",
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
        "--tag", f"epoch16_{stage}_{clusterer}",
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
            log_tail = (output / "run.log").read_text(
                errors="replace"
            ).splitlines()[-10:]
        except OSError:
            log_tail = []
        if log_tail:
            print("--- failing point log tail ---", file=sys.stderr)
            print("\n".join(log_tail), file=sys.stderr)
            print("------------------------------", file=sys.stderr)
        raise RuntimeError(
            f"point failed ({clusterer}, tbeta={tbeta}, td={td}); "
            f"see {output / 'run.log'}"
        )
    if not _complete(output, args.provenance_sha256, config):
        raise RuntimeError(
            f"point wrote incomplete or mismatched provenance: {output}"
        )
    print(
        f"[{stage}] {clusterer:<18} tb={tbeta:<5g} td={td:<5g} "
        f"mh={min_hits} m={merge_td:g} a={attach_td:g} "
        f"{time.time() - started:.1f}s",
        flush=True,
    )
    return output


def _load_base_points(
    root: Path, stage: str = "full_grid"
) -> list[tuple[Path, dict]]:
    points = []
    for path in sorted((root / stage).glob("*/*/summary.json")):
        summary = json.loads(path.read_text())
        points.append((path.parent, summary))
    return points


def _pareto_pair(points, x_getter, y_getter):
    points = [
        item for item in points
        if item[1].get("ggtf_n_matched_gt10", 0) > 0
        and item[1].get("promotable_operating_point", True)
    ]
    result = []
    for path, summary in points:
        x = x_getter(summary)
        y = y_getter(summary)
        dominated = any(
            x_getter(other) <= x
            and y_getter(other) >= y
            and (
                x_getter(other) < x
                or y_getter(other) > y
            )
            for _, other in points
        )
        if not dominated:
            result.append((path, summary))
    return result


def _pareto(points, y_getter):
    return _pareto_pair(
        points, lambda summary: summary["ggtf_fake_rate_gt10"], y_getter
    )


def _refinement_bases(
    root: Path,
    stage: str = "full_grid",
    cap: int = 12,
) -> list[tuple[Path, dict]]:
    points = _load_base_points(root, stage)
    fronts = []
    getters = [
        lambda s: s["idea"]["match_rate"],
        lambda s: s["idea"]["def2"],
        lambda s: s["idea"]["efficiency"],
        lambda s: s["low_pt_idea"]["0.1-0.2"]["match_rate"],
        lambda s: s["low_pt_idea"]["0.1-0.2"]["def2"],
        lambda s: s["low_pt_idea"]["0.2-0.5"]["match_rate"],
        lambda s: s["low_pt_idea"]["0.2-0.5"]["def2"],
    ]
    for getter in getters:
        fronts.extend(_pareto(points, getter))
    matched = lambda s: max(s["ggtf_n_matched_gt10"], 1)
    for cost in (
        lambda s: s["ggtf_n_fake_clone_gt10"] / matched(s),
        lambda s: s["ggtf_n_fake_spurious_gt10"] / matched(s),
        lambda s: s["candidates_per_event"],
    ):
        fronts.extend(
            _pareto_pair(points, cost, lambda s: s["idea"]["def2"])
        )
    unique = {str(path): (path, summary) for path, summary in fronts}
    selected = sorted(
        unique.values(), key=lambda item: item[1]["ggtf_fake_rate_gt10"]
    )
    if len(selected) > cap:
        indices = {
            round(index * (len(selected) - 1) / (cap - 1))
            for index in range(cap)
        }
        selected = [selected[index] for index in sorted(indices)]
    return selected


def _file_identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256(path),
    }


def _preflight(args) -> dict:
    manifest = json.loads((args.cache / "manifest.json").read_text())
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in EXPECTED_CACHE_FIELDS.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"cache manifest mismatch: {mismatches}")
    checkpoint = Path(manifest["checkpoint"])
    checkpoint_sha = _sha256(checkpoint)
    if checkpoint_sha != manifest["checkpoint_sha256"]:
        raise RuntimeError("immutable checkpoint hash no longer matches cache manifest")
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise RuntimeError(
            "cache checkpoint is not the explicitly pinned epoch-16 checkpoint"
        )
    for path in (args.cache / "forward_hits.parquet", args.mc_signal):
        if not path.exists():
            raise FileNotFoundError(path)
    hits = pl.read_parquet(
        args.cache / "forward_hits.parquet",
        columns=["seed", "event_id", "mc_index"],
    )
    actual = {
        "seeds": sorted(hits["seed"].unique().to_list()),
        "events": hits.select(["seed", "event_id"]).unique().height,
        "particle_zero_rows": int((hits["mc_index"] == 0).sum()),
        "particle_zero_events": hits.filter(
            pl.col("mc_index") == 0
        ).select(["seed", "event_id"]).unique().height,
    }
    if actual["seeds"] != [181] or actual["events"] != 500:
        raise RuntimeError(f"cache row scope mismatch: {actual}")
    if actual["particle_zero_rows"] <= 0:
        raise RuntimeError("particle zero is absent from cache rows")
    mc = pl.read_parquet(
        args.mc_signal, columns=["seed", "event_id", "mc_index"]
    )
    duplicates = (
        mc.group_by(["seed", "event_id", "mc_index"])
        .len()
        .filter(pl.col("len") != 1)
    )
    if len(duplicates):
        raise RuntimeError(f"mc_signal contains {len(duplicates)} duplicate keys")
    if sorted(mc["seed"].unique().to_list()) != [181]:
        raise RuntimeError("mc_signal is not scoped exactly to seed 181")
    dataset_manifest = json.loads(args.dataset_manifest.read_text())
    if (
        dataset_manifest.get("truth_policy") != "no-keep-all"
        or dataset_manifest.get("validation_seeds") != "181-190"
    ):
        raise RuntimeError("dataset manifest is not the final no-keep-all split")
    data_root = Path(dataset_manifest["data_dir"])
    seed_entries = [
        entry for entry in dataset_manifest["files"]
        if entry["path"].startswith("seed_181/")
    ]
    if len(seed_entries) != 3:
        raise RuntimeError("dataset manifest lacks the three seed-181 files")
    for entry in seed_entries:
        stat = (data_root / entry["path"]).stat()
        if (
            stat.st_size != entry["size"]
            or stat.st_mtime_ns != entry["mtime_ns"]
        ):
            raise RuntimeError(
                f"seed-181 raw data changed: {entry['path']}"
            )
    payload = {
        "schema_version": 2,
        "scope": (
            "500 no-keep-all validation events from seed 181; no keep-all or "
            "Loopers input is configured"
        ),
        "cache_manifest": manifest,
        "dataset_manifest_claims": {
            key: dataset_manifest[key]
            for key in (
                "fingerprint", "train_seeds", "validation_seeds",
                "truth_policy", "data_dir",
            )
        },
        "actual_cache_scope": actual,
        "inputs": {
            "forward_cache": _file_identity(
                args.cache / "forward_hits.parquet"
            ),
            "mc_signal": _file_identity(args.mc_signal),
            "dataset_manifest": _file_identity(args.dataset_manifest),
            "checkpoint": _file_identity(checkpoint),
        },
        "source_sha256": _source_hashes(args.package),
        "runtime": {
            "python": sys.version,
            **{
                package: importlib.metadata.version(package)
                for package in (
                    "numpy", "polars", "scipy", "scikit-learn", "hdbscan"
                )
            },
        },
        "grid": {
            "tbeta": list(TBETAS),
            "td": list(TDS),
            "full_grid_clusterers": ["greedy", "self_seed_greedy"],
            "density_clusterers": ["dbscan", "hdbscan"],
            "min_cluster_hits": 4,
            "min_target_hits": 3,
        },
    }
    campaign_path = args.output / "campaign_manifest.json"
    campaign_sha = _canonical_hash(payload)
    campaign = {**payload, "campaign_sha256": campaign_sha}
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--mc-signal", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.output / "campaign.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another sweep process holds the campaign lock") from exc
    args.campaign = _preflight(args)
    args.provenance_sha256 = args.campaign["campaign_sha256"]

    for clusterer in ("greedy", "self_seed_greedy"):
        for tbeta in TBETAS:
            for td in TDS:
                _run_point(args, "full_grid", clusterer, tbeta, td)

    # Density methods do not use beta; scan each distance scale exactly once.
    for clusterer in ("dbscan", "hdbscan"):
        for td in TDS:
            _run_point(args, "density", clusterer, 0.5, td)

    bases = _refinement_bases(args.output)
    provenance = []
    for base_path, summary in bases:
        clusterer = summary["clusterer"]
        tbeta = summary["tbeta"]
        td = summary["td"]
        provenance.append({
            "source": str(base_path),
            "clusterer": clusterer,
            "tbeta": tbeta,
            "td": td,
        })
        for min_hits, merge, attach in (
            (3, 0.0, 0.0),
            (5, 0.0, 0.0),
            (4, 0.5 * td, 0.0),
            (4, td, 0.0),
            (4, 0.0, 0.5 * td),
            (4, 0.0, td),
            (4, 0.5 * td, 0.5 * td),
        ):
            _run_point(
                args, "refinement", clusterer, tbeta, td,
                min_hits=min_hits, merge_td=merge, attach_td=attach,
            )
    (args.output / "refinement_sources.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    density_bases = _refinement_bases(
        args.output, stage="density", cap=6
    )
    density_provenance = []
    for base_path, summary in density_bases:
        clusterer = summary["clusterer"]
        td = summary["td"]
        density_provenance.append({
            "source": str(base_path),
            "clusterer": clusterer,
            "tbeta": 0.5,
            "td": td,
        })
        for min_hits in (3, 5):
            _run_point(
                args, "density_refinement", clusterer, 0.5, td,
                min_hits=min_hits,
            )
    (args.output / "density_refinement_sources.json").write_text(
        json.dumps(density_provenance, indent=2) + "\n"
    )
    _assert_campaign_current(args)
    (args.output / "SWEEP_COMPLETE").write_text(
        json.dumps({
            "campaign_sha256": args.provenance_sha256,
            "completed_at_unix": time.time(),
        }, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
