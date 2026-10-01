#!/usr/bin/env python3
"""Generate machine-verifiable evidence consumed by the epoch-16 audit report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


PYTHON_FILES = (
    "src/eval/fcc_cache_parallel.py",
    "src/eval/forward_pass.py",
    "src/eval/merge_forward_shards.py",
    "src/eval/plot_fcc_metrics.py",
    "src/eval/report_epoch16_cluster_sweep.py",
    "src/eval/run_epoch16_cluster_sweep.py",
    "src/eval/run_epoch16_oracle_appendix.py",
    "src/eval/audit_epoch16_model_stack.py",
    "src/eval/audit_epoch16_cached_embeddings.py",
    "src/eval/audit_epoch16_data_sampler.py",
    "src/eval/attach_positions.py",
    "src/eval/build_epoch16_audit_report.py",
    "src/eval/verify_epoch16_audit.py",
)
TEST_FILES = (
    "src/eval/test_metric_semantics.py",
    "src/eval/test_forward_manifest.py",
    "src/eval/test_select_milestone_checkpoint.py",
    "src/eval/test_paper_cga.py",
    "src/cgatr/tests/test_cga.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], cwd: Path, env: dict | None = None) -> dict:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "passed": result.returncode == 0,
        "output_tail": result.stdout[-6000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(args.package),
        "CUDA_VISIBLE_DEVICES": "",
        "FIX_PARTICLE_ZERO": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    compile_result = _run(
        [sys.executable, "-m", "py_compile", *PYTHON_FILES],
        args.package,
        env,
    )
    test_result = _run(
        [sys.executable, "-m", "pytest", *TEST_FILES, "-q"],
        args.package,
        env,
    )
    smoke_result = _run(
        [
            sys.executable,
            "src/eval/smoke_train_configs.py",
            "--only",
            "final corrected E3",
        ],
        args.package,
        env,
    )
    service_result = _run(
        [
            "systemctl", "--user", "show",
            "cgatr-final-hail-mary.service",
            "-p", "ActiveState", "-p", "SubState", "-p", "NRestarts",
        ],
        args.package,
        env,
    )
    service_values = dict(
        line.split("=", 1)
        for line in service_result["output_tail"].splitlines()
        if "=" in line
    )
    service_result["passed"] = (
        service_result["returncode"] == 0
        and service_values.get("ActiveState") == "active"
        and service_values.get("SubState") == "running"
    )
    service_result["values"] = service_values

    run_manifest = json.loads(args.run_manifest.read_text())
    locked_sources = {
        relative: {
            "expected": expected,
            "actual": _sha256(args.package / relative),
        }
        for relative, expected in run_manifest["source_sha256"].items()
    }
    locked_sources_pass = all(
        item["expected"] == item["actual"]
        for item in locked_sources.values()
    )
    sweep_summary = json.loads(
        (args.sweep_root / "summary.json").read_text()
    )
    campaign = json.loads(
        (args.sweep_root / "campaign_manifest.json").read_text()
    )
    configured_inputs = [
        item["path"] for item in campaign["inputs"].values()
    ]
    forbidden_tokens = ("keepall", "keep-all", "loopers")
    configured_holdout_inputs = [
        path for path in configured_inputs
        if any(token in path.lower() for token in forbidden_tokens)
    ]
    required_audits = [
        args.audit_root / "model_stack.json",
        args.audit_root / "data_sampler.json",
        args.audit_root / "embedding_loss/embedding_loss_audit.json",
        args.audit_root / "training_metric_op_mh1/summary.json",
    ]
    artifacts_present = all(path.is_file() for path in required_audits)
    audit_artifact_sha256 = {
        str(path.resolve()): _sha256(path)
        for path in required_audits if path.is_file()
    }
    audit_source_sha256 = {
        relative: _sha256(args.package / relative)
        for relative in (
            "src/eval/audit_epoch16_model_stack.py",
            "src/eval/audit_epoch16_cached_embeddings.py",
            "src/eval/audit_epoch16_data_sampler.py",
            "src/eval/build_epoch16_audit_report.py",
        )
    }
    checks = {
        "compile": compile_result,
        "focused_pytest": test_result,
        "final_configuration_smoke": smoke_result,
        "training_service": service_result,
        "locked_training_sources": {
            "passed": locked_sources_pass,
            "files": locked_sources,
        },
        "sweep_completion": {
            "passed": (
                sweep_summary.get("exact_configuration_set_verified") is True
                and sweep_summary.get("campaign_sha256")
                == campaign.get("campaign_sha256")
            ),
            "campaign_sha256": campaign.get("campaign_sha256"),
            "points": sweep_summary.get("points"),
        },
        "audit_artifacts": {
            "passed": artifacts_present,
            "paths": [str(path.resolve()) for path in required_audits],
            "artifact_sha256": audit_artifact_sha256,
            "source_sha256": audit_source_sha256,
        },
        "evidence_boundary": {
            "passed": not configured_holdout_inputs,
            "claim": (
                "No keep-all or Loopers path is configured in the pinned "
                "campaign inputs; this is configuration evidence, not an "
                "operating-system file-access audit."
            ),
            "configured_inputs": configured_inputs,
            "configured_holdout_inputs": configured_holdout_inputs,
        },
    }
    overall_passed = all(check["passed"] for check in checks.values())
    report = {
        "schema_version": 1,
        "overall_passed": overall_passed,
        "verification_source_sha256": _sha256(Path(__file__)),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "overall_passed": overall_passed,
        "checks": {
            key: value["passed"] for key, value in checks.items()
        },
    }, indent=2))
    if not overall_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
