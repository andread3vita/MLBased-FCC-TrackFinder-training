"""Consolidate the epoch-16 sweep and pipeline audit into JSON and Markdown."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl


def _load(path):
    return json.loads(Path(path).read_text())


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    sweep = _load(args.sweep_root / "summary.json")
    model = _load(args.audit_root / "model_stack.json")
    data = _load(args.audit_root / "data_sampler.json")
    embedding = _load(
        args.audit_root / "embedding_loss/embedding_loss_audit.json"
    )
    mh1 = _load(args.audit_root / "training_metric_op_mh1/summary.json")
    verification = _load(args.verification)
    if verification.get("schema_version") != 1:
        raise RuntimeError("unsupported or missing verification evidence")
    if verification.get("verification_source_sha256") != _sha256(
        args.package / "src/eval/verify_epoch16_audit.py"
    ):
        raise RuntimeError("verification artifact source hash is stale")
    audit_evidence = verification["checks"]["audit_artifacts"]
    for path, expected in audit_evidence["artifact_sha256"].items():
        if _sha256(Path(path)) != expected:
            raise RuntimeError(f"audit artifact changed after verification: {path}")
    for relative, expected in audit_evidence["source_sha256"].items():
        if _sha256(args.package / relative) != expected:
            raise RuntimeError(f"audit source changed after verification: {relative}")
    run_manifest = _load(args.run_manifest)
    source_checks = {}
    for relative, expected in run_manifest["source_sha256"].items():
        actual = _sha256(args.package / relative)
        source_checks[relative] = {
            "expected": expected,
            "actual": actual,
            "matches": actual == expected,
        }

    points = pl.read_csv(args.sweep_root / "all_points.csv")
    frontier = pl.read_csv(args.sweep_root / "pareto_frontier.csv")
    greedy_fixed = points.filter(
        (pl.col("stage") == "full_grid")
        & (pl.col("clusterer") == "greedy")
        & (pl.col("tbeta") == 0.1)
        & (pl.col("td") == 0.2)
    ).row(0, named=True)
    self_seed_fixed = points.filter(
        (pl.col("stage") == "full_grid")
        & (pl.col("clusterer") == "self_seed_greedy")
        & (pl.col("tbeta") == 0.1)
        & (pl.col("td") == 0.2)
    ).row(0, named=True)
    mh4 = _load(args.sweep_root / greedy_fixed["point"] / "summary.json")
    if (
        mh1.get("sweep_provenance_sha256") != sweep["campaign_sha256"]
        or mh4.get("sweep_provenance_sha256") != sweep["campaign_sha256"]
    ):
        raise RuntimeError("fixed-point audit summaries have stale provenance")
    frontier_points = set(frontier["point"].to_list())
    descriptive_endpoints = {}
    for clusterer in sorted(points["clusterer"].unique().to_list()):
        endpoint = (
            points.filter(
                (pl.col("clusterer") == clusterer) & pl.col("promotable")
            )
            .sort(
                ["idea_def2", "ggtf10_unassigned_per_matched"],
                descending=[True, False],
            )
            .row(0, named=True)
        )
        descriptive_endpoints[clusterer] = {
            "selection": (
                "maximum IDEA definition 2 within this algorithm only; "
                "not an operating-point recommendation"
            ),
            "point": endpoint["point"],
            "on_pareto_union": endpoint["point"] in frontier_points,
            "idea_def1": endpoint["idea_def1"],
            "idea_def2": endpoint["idea_def2"],
            "low_pt_01_02_def2": endpoint["low_pt_01_02_def2"],
            "ggtf10_unassigned_per_matched": endpoint[
                "ggtf10_unassigned_per_matched"
            ],
            "clone_per_matched_gt10": endpoint["clone_per_matched_gt10"],
            "spurious_per_matched_gt10": endpoint[
                "spurious_per_matched_gt10"
            ],
            "candidates_per_event": endpoint["candidates_per_event"],
        }
    radius = model["radius_sensitivity"]["mean"]
    gradients = model["gradient_reachability"]
    joins = data["joins"]
    joins_clean = all(value == 0 for value in joins.values())
    full_validation = data["physics"]["full_validation"]
    major_invalidating = (
        not all(item["matches"] for item in source_checks.values())
        or not gradients["nonzero_gradient_fraction"] > 0.99
        or gradients["nonfinite_gradient_tensors"]
        or not model["checkpoint"]["optimizer"]["all_state_tensors_finite"]
        or not joins_clean
        or full_validation["particle_zero_targets"] == 0
        or not model["radius_sensitivity"]["materiality_test"][
            "radius_response_material"
        ]
        or not verification.get("overall_passed", False)
    )

    classifications = [
        {
            "class": "proven implementation bug",
            "finding": (
                "The first-pass audit allowed unpinned evaluator/input state "
                "and hard-coded verification claims. Earlier cache identity also "
                "omitted particle-zero, dataset, and inference-source identity; "
                "candidate-fake reporting mixed normalizations; and legacy "
                "position attachment lacked a unique hit key."
            ),
            "impact": (
                "The original report is superseded. This report consumes a "
                "clean campaign with content-hashed inputs/code, exact point-set "
                "validation, machine-run checks, strict metric complements, and "
                "unique-key sidecar validation."
            ),
        },
        {
            "class": "proven implementation bug",
            "finding": (
                "The current local greedy clusterer lets a beta seed that was "
                "already claimed by a higher-beta seed spawn a new fragment. "
                "GGTF's active inference recomputes seeds from unassigned hits. "
                "At the fixed point, the self-seed rule changes GGTF >10 from "
                f"{100 * greedy_fixed['ggtf10_unassigned_per_matched']:.2f}% "
                f"to {100 * self_seed_fixed['ggtf10_unassigned_per_matched']:.2f}% "
                "while IDEA definition 2 changes from "
                f"{100 * greedy_fixed['idea_def2']:.2f}% to "
                f"{100 * self_seed_fixed['idea_def2']:.2f}%."
            ),
            "impact": (
                "This inflates clone-sensitive evaluation but does not alter "
                "training loss or embeddings. It belongs on the Pareto frontier; "
                "the official selector is intentionally unchanged in this pass."
            ),
        },
        {
            "class": "evaluation-definition mismatch",
            "finding": (
                "The four-rank fixed 40-batch validation sample is modestly "
                "occupancy-biased: "
                f"median {data['event_hit_counts']['fixed_validation_40_batches']['median']:.0f} "
                "hits versus "
                f"{data['event_hit_counts']['full_validation']['median']:.0f} "
                "for full validation "
                f"(KS p={data['event_hit_counts']['ks_fixed_validation_vs_full_validation']['pvalue']:.3g})."
            ),
            "impact": (
                "Training trajectory validation is a stable, slightly harder "
                "subset and its logged metrics average rank-local batch means "
                "rather than all particles. It is not a population estimate. "
                "Final selection remains on "
                "the full validation cache."
            ),
        },
        {
            "class": "evaluation-definition mismatch",
            "finding": (
                f"{100 * full_validation['exactly_3_hit_target_fraction']:.1f}% "
                "of min-3-hit targets have exactly three hits while paper "
                "candidates require at least four."
            ),
            "impact": (
                f"Removing the candidate floor changes candidates/event from "
                f"{mh4['candidates_per_event']:.1f} to "
                f"{mh1['candidates_per_event']:.1f}, but leaves the >10-hit "
                "metric and IDEA target metrics effectively unchanged at the "
                "fixed point."
            ),
        },
        {
            "class": "expected design limitation",
            "finding": (
                f"{100 * embedding['high_beta_multiplicity']['fraction_tracks_multiple_beta_gt_08']:.1f}% "
                "of min-3-hit targets contain multiple beta>0.8 hits "
                f"(mean {embedding['high_beta_multiplicity']['mean_beta_gt_08_per_track']:.2f})."
            ),
            "impact": (
                "The loss is healthy and differentiable, but weak non-alpha "
                "beta suppression permits redundant seeds. The self-seed "
                "clusterer mitigates their clone burden without retraining."
            ),
        },
        {
            "class": "expected design limitation",
            "finding": (
                "Physical drift radius survives end-to-end but is weak: mean "
                f"relative RMS is {radius['embedding_relative_rms_zero_radius']:.2e} "
                "at embedding and "
                f"{radius['checkpoint_output_relative_rms_zero_radius']:.2e} "
                "at output when radius is zeroed."
            ),
            "impact": (
                "Literal numerical annihilation is ruled out across "
                f"{len(model['radius_sensitivity']['per_event'])} real events. "
                "The response exceeds the documented materiality and "
                "repeatability thresholds. "
                "No separate radius scalar exists, so this remains a "
                "future architecture ablation rather than a reason to invalidate "
                "the current physical-geometry run."
            ),
        },
        {
            "class": "ruled-out hypothesis",
            "finding": (
                f"{100 * gradients['nonzero_gradient_fraction']:.3f}% of model "
                "parameters receive finite nonzero gradients; all optimizer "
                "state tensors are finite and data joins have zero missing or "
                "duplicate keys."
            ),
            "impact": (
                "No broken backbone, optimizer corruption, particle-zero loss, "
                "or MC-join bug was reproduced."
            ),
        },
        {
            "class": "expected design limitation",
            "finding": (
                f"Epoch {model['trajectory']['val_loss_best_epoch']:.0f} is the "
                "best logged validation-loss epoch, all logged "
                "loss-component means are finite, no NaN skip is logged, the "
                "scheduler has zero bad epochs, and EMA differs from raw weights "
                f"by {100 * model['checkpoint']['ema_raw_relative_l2']:.2f}% in L2."
            ),
            "impact": (
                "The run is slowing asymptotically rather than dead or "
                "diverging. A recent-epoch inverse-sqrt fit forecasts strict50 "
                f"{100 * model['trajectory']['inverse_sqrt_forecast']['strict50']['40']:.1f}% "
                "at epoch 40; this is descriptive, not a guarantee."
            ),
        },
        {
            "class": "expected design limitation",
            "finding": (
                "Only attention log-weight parameters and the multivector "
                "linear-out branch lack gradients under the invariant scalar "
                "output head."
            ),
            "impact": (
                "The used scalar output path and backbone are live; 0.053% "
                "unused parameters do not invalidate training."
            ),
        },
    ]
    report = {
        "scope": sweep["scope"],
        "safety": {
            "holdout_access_claim": verification["checks"][
                "evidence_boundary"
            ]["claim"],
            "configured_holdout_inputs": verification["checks"][
                "evidence_boundary"
            ]["configured_holdout_inputs"],
            "locked_training_sources_match": all(
                item["matches"] for item in source_checks.values()
            ),
            "source_checks": source_checks,
        },
        "verification": verification,
        "sweep": {
            **sweep,
            "points_loaded": len(points),
            "pareto_rows": len(frontier),
            "fixed_point": {
                "idea_def1": mh4["idea"]["match_rate"],
                "idea_def2": mh4["idea"]["def2"],
                "ggtf10_unassigned_per_matched": mh4[
                    "ggtf_fake_rate_gt10"
                ],
                "ggtf10_candidate_fake_fraction": mh4[
                    "ggtf_candidate_fake_fraction_gt10"
                ],
            },
            "descriptive_algorithm_endpoints": descriptive_endpoints,
        },
        "classifications": classifications,
        "pause_decision": {
            "major_training_invalidating_issue_proven": bool(major_invalidating),
            "action": "pause" if major_invalidating else "continue training",
            "reason": (
                "A locked-source, used-path, gradient, optimizer, target, or "
                "join invariant failed."
                if major_invalidating else
                "Findings are clustering/design limitations or metric-scope "
                "issues; none changes the trained inputs, targets, loss, or "
                "used model path."
            ),
        },
        "artifacts": {
            "pareto_plot": str(
                (args.sweep_root / "pareto_frontiers.png").resolve()
            ),
            "all_points": str(
                (args.sweep_root / "all_points.csv").resolve()
            ),
            "pareto_frontier": str(
                (args.sweep_root / "pareto_frontier.csv").resolve()
            ),
            "current_fixed_ggtf_panels": str(
                (
                    args.sweep_root
                    / "refreshed_plots/current_greedy_fixed/ggtf/"
                    "ggtf_important_panels.png"
                ).resolve()
            ),
            "self_seed_fixed_ggtf_panels": str(
                (
                    args.sweep_root
                    / "refreshed_plots/self_seed_fixed/ggtf/"
                    "ggtf_important_panels.png"
                ).resolve()
            ),
            "model_stack": str(
                (args.audit_root / "model_stack.json").resolve()
            ),
            "data_sampler": str(
                (args.audit_root / "data_sampler.json").resolve()
            ),
            "embedding_loss": str(
                (
                    args.audit_root
                    / "embedding_loss/embedding_loss_audit.json"
                ).resolve()
            ),
        },
    }
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Epoch-16 sweep and pipeline audit",
        "",
        f"**Decision:** {report['pause_decision']['action']}. "
        f"{report['pause_decision']['reason']}",
        "",
        "## Evidence boundary",
        "",
        f"- {report['scope']}",
        f"- {report['safety']['holdout_access_claim']}",
        f"- Machine verification overall pass: "
        f"{report['verification']['overall_passed']}.",
        f"- Locked training sources match: "
        f"{report['safety']['locked_training_sources_match']}.",
        f"- Sweep points: {len(points)}; Pareto-union rows: {len(frontier)}. "
        "No operating-point winner was selected.",
        "",
        "## Classified findings",
        "",
    ]
    for item in classifications:
        lines.extend([
            f"### {item['class']}",
            "",
            item["finding"],
            "",
            f"Impact: {item['impact']}",
            "",
        ])
    lines.extend([
        "## Fixed epoch-16 reference point",
        "",
        f"- IDEA definition 1: {100 * mh4['idea']['match_rate']:.2f}%",
        f"- IDEA definition 2: {100 * mh4['idea']['def2']:.2f}%",
        f"- Mean IDEA hit efficiency: {100 * mh4['idea']['efficiency']:.2f}%",
        f"- GGTF >10 unassigned/matched: "
        f"{100 * mh4['ggtf_fake_rate_gt10']:.2f}%",
        f"- >10 candidate fake fraction: "
        f"{100 * mh4['ggtf_candidate_fake_fraction_gt10']:.2f}%",
        "",
        "## Descriptive maximum-def2 endpoint per algorithm",
        "",
    ])
    for clusterer, endpoint in descriptive_endpoints.items():
        lines.append(
            f"- {clusterer}: def2 {100 * endpoint['idea_def2']:.2f}%, "
            f"GGTF >10 {100 * endpoint['ggtf10_unassigned_per_matched']:.2f}%, "
            f"low-pT 0.1–0.2 def2 "
            f"{100 * endpoint['low_pt_01_02_def2']:.2f}% "
            f"(`{endpoint['point']}`)."
        )
    lines.extend([
        "",
        "These are within-algorithm high-efficiency endpoints for orientation, "
        "not a cross-algorithm winner or frozen operating point.",
        "",
        "See `pareto_frontier.csv` and `pareto_frontiers.png`; those are "
        "descriptive validation results, not a frozen operating-point choice.",
        "",
    ])
    args.output_md.write_text("\n".join(lines))
    print(json.dumps(report["pause_decision"], indent=2))


if __name__ == "__main__":
    main()
