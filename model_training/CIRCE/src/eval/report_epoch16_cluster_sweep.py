"""Build Pareto tables and plots from the epoch-16 CPU clustering study."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.eval.run_epoch16_cluster_sweep import (
    TBETAS,
    TDS,
    _canonical_hash,
    _sha256,
    _source_hashes,
)


Y_METRICS = {
    "idea_def1": "IDEA definition 1",
    "idea_def2": "IDEA definition 2",
    "low_pt_01_02_def1": "IDEA def1, 0.1–0.2 GeV",
    "low_pt_01_02_def2": "IDEA def2, 0.1–0.2 GeV",
    "low_pt_02_05_def1": "IDEA def1, 0.2–0.5 GeV",
    "low_pt_02_05_def2": "IDEA def2, 0.2–0.5 GeV",
    "idea_hit_eff": "IDEA mean hit efficiency",
}
COST_METRICS = {
    "clone_per_matched_gt10": "Clone candidates / matched [%]",
    "spurious_per_matched_gt10": "Spurious candidates / matched [%]",
    "candidates_per_event": "Candidates / event",
}


def _rate(summary: dict, split: str, *keys: str) -> float:
    node = summary if split == "full" else summary["event_splits"][split]
    for key in keys:
        node = node[key]
    return float(node)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.inf


def _safe_delta(left: float, right: float) -> float | None:
    if not math.isfinite(left) or not math.isfinite(right):
        return None
    return abs(left - right)


def _flatten(root: Path, path: Path) -> dict:
    summary = json.loads(path.read_text())
    relative = path.parent.relative_to(root)
    parts = relative.parts
    stage = parts[0]
    clusterer = summary["clusterer"]
    matched = summary["ggtf_n_matched_gt10"]
    row = {
        "point": str(relative),
        "stage": stage,
        "clusterer": clusterer,
        "tbeta": summary["tbeta"],
        "td": summary["td"],
        "min_cluster_hits": summary["min_cluster_hits"],
        "merge_td": summary["merge_td"],
        "attach_td": summary["attach_td"],
        "helix_tol": summary.get("helix_tol", 0.0),
        "promotable": summary.get("promotable_operating_point", True),
        "sweep_provenance_sha256": summary.get(
            "sweep_provenance_sha256"
        ),
        "n_events": summary["n_events"],
        "idea_def1": summary["idea"]["match_rate"],
        "idea_def2": summary["idea"]["def2"],
        "idea_hit_eff": summary["idea"]["efficiency"],
        "all_target_def1": summary["no_cuts"]["match_rate"],
        "all_target_def2": summary["no_cuts"]["def2"],
        "min3_target_def1": summary[
            "min3_targets_no_reconstruction_cuts"
        ]["match_rate"],
        "min3_target_def2": summary[
            "min3_targets_no_reconstruction_cuts"
        ]["def2"],
        "low_pt_01_02_def1": summary["low_pt_idea"]["0.1-0.2"]["match_rate"],
        "low_pt_01_02_def2": summary["low_pt_idea"]["0.1-0.2"]["def2"],
        "low_pt_02_05_def1": summary["low_pt_idea"]["0.2-0.5"]["match_rate"],
        "low_pt_02_05_def2": summary["low_pt_idea"]["0.2-0.5"]["def2"],
        "ggtf10_unassigned_per_matched": summary["ggtf_fake_rate_gt10"],
        "candidate_fake_fraction_gt10": summary[
            "ggtf_candidate_fake_fraction_gt10"
        ],
        "n_candidates_gt10": summary["ggtf_n_cand_gt10"],
        "n_matched_gt10": summary["ggtf_n_matched_gt10"],
        "clone_per_matched_gt10": _ratio(
            summary["ggtf_n_fake_clone_gt10"], matched
        ),
        "spurious_per_matched_gt10": _ratio(
            summary["ggtf_n_fake_spurious_gt10"], matched
        ),
        "candidates_per_event": summary["candidates_per_event"],
    }
    for split in ("even", "odd"):
        split_node = summary["event_splits"][split]
        split_matched = split_node["ggtf_n_matched_gt10"]
        row.update({
            f"{split}_idea_def1": split_node["idea"]["match_rate"],
            f"{split}_idea_def2": split_node["idea"]["def2"],
            f"{split}_idea_hit_eff": split_node["idea"]["efficiency"],
            f"{split}_low_pt_01_02_def1": (
                split_node["low_pt_idea"]["0.1-0.2"]["match_rate"]
            ),
            f"{split}_low_pt_01_02_def2": (
                split_node["low_pt_idea"]["0.1-0.2"]["def2"]
            ),
            f"{split}_low_pt_02_05_def1": (
                split_node["low_pt_idea"]["0.2-0.5"]["match_rate"]
            ),
            f"{split}_low_pt_02_05_def2": (
                split_node["low_pt_idea"]["0.2-0.5"]["def2"]
            ),
            f"{split}_ggtf10": split_node["ggtf_fake_rate_gt10"],
            f"{split}_clone_gt10": (
                _ratio(split_node["ggtf_n_fake_clone_gt10"], split_matched)
            ),
            f"{split}_spurious_gt10": (
                _ratio(split_node["ggtf_n_fake_spurious_gt10"], split_matched)
            ),
        })
    row["split_delta_idea_def2"] = abs(
        row["even_idea_def2"] - row["odd_idea_def2"]
    )
    row["split_delta_idea_def1"] = abs(
        row["even_idea_def1"] - row["odd_idea_def1"]
    )
    row["split_delta_idea_hit_eff"] = abs(
        row["even_idea_hit_eff"] - row["odd_idea_hit_eff"]
    )
    row["split_delta_low_pt_01_02_def2"] = abs(
        row["even_low_pt_01_02_def2"] - row["odd_low_pt_01_02_def2"]
    )
    row["split_delta_low_pt_01_02_def1"] = abs(
        row["even_low_pt_01_02_def1"] - row["odd_low_pt_01_02_def1"]
    )
    row["split_delta_low_pt_02_05_def1"] = abs(
        row["even_low_pt_02_05_def1"] - row["odd_low_pt_02_05_def1"]
    )
    row["split_delta_low_pt_02_05_def2"] = abs(
        row["even_low_pt_02_05_def2"] - row["odd_low_pt_02_05_def2"]
    )
    row["split_delta_ggtf10"] = _safe_delta(
        row["even_ggtf10"], row["odd_ggtf10"]
    )
    return row


def _pareto_pair(rows: list[dict], x_key: str, y_key: str) -> set[str]:
    result = set()
    valid = [
        row for row in rows
        if row["n_matched_gt10"] > 0 and row["promotable"]
    ]
    for row in valid:
        dominated = any(
            other[x_key] <= row[x_key]
            and other[y_key] >= row[y_key]
            and (other[x_key] < row[x_key] or other[y_key] > row[y_key])
            for other in valid
        )
        if not dominated:
            result.add(row["point"])
    return result


def _pareto(rows: list[dict], y_key: str) -> set[str]:
    return _pareto_pair(
        rows, "ggtf10_unassigned_per_matched", y_key
    )


def _heatmap(rows, clusterer, key, label, output):
    selected = [
        row for row in rows
        if row["stage"] == "full_grid" and row["clusterer"] == clusterer
    ]
    tbeta = sorted({row["tbeta"] for row in selected})
    td = sorted({row["td"] for row in selected})
    lookup = {(row["tbeta"], row["td"]): row[key] for row in selected}
    values = np.asarray([[lookup[(b, d)] for d in td] for b in tbeta])
    fig, ax = plt.subplots(figsize=(9.2, 4.7))
    image = ax.imshow(values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(td)), [f"{value:g}" for value in td], rotation=45)
    ax.set_yticks(range(len(tbeta)), [f"{value:g}" for value in tbeta])
    ax.set_xlabel("distance threshold $t_d$")
    ax.set_ylabel("beta threshold $t_\\beta$")
    ax.set_title(f"{clusterer}: {label}")
    fig.colorbar(image, ax=ax, label="rate")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _frontier_plot(rows, fronts, output):
    colors = {
        "greedy": "#2369bd",
        "self_seed_greedy": "#d55e00",
        "dbscan": "#009e73",
        "hdbscan": "#8b5cf6",
    }
    fig, axes = plt.subplots(3, 3, figsize=(15.0, 12.4))
    for ax, (key, label) in zip(axes.flat, Y_METRICS.items()):
        for clusterer, color in colors.items():
            selected = [
                row for row in rows
                if row["clusterer"] == clusterer
                and row["n_matched_gt10"] > 0
                and row["promotable"]
            ]
            if not selected:
                continue
            ax.scatter(
                [
                    max(
                        1e-4,
                        100 * row["ggtf10_unassigned_per_matched"],
                    )
                    for row in selected
                ],
                [100 * row[key] for row in selected],
                s=12, alpha=0.28, color=color, label=clusterer,
            )
        front_rows = [row for row in rows if row["point"] in fronts[key]]
        front_rows.sort(key=lambda row: row["ggtf10_unassigned_per_matched"])
        ax.plot(
            [
                max(1e-4, 100 * row["ggtf10_unassigned_per_matched"])
                for row in front_rows
            ],
            [100 * row[key] for row in front_rows],
            "k.--", linewidth=1.0, markersize=5, label="Pareto",
        )
        ax.set_xscale("log")
        ax.set_xlabel("GGTF >10 unassigned / matched [%]")
        ax.set_ylabel(f"{label} [%]")
        ax.grid(alpha=0.2, which="both")
    axes.flat[0].legend(fontsize=8)
    for ax in axes.flat[len(Y_METRICS):]:
        ax.axis("off")
    fig.suptitle(
        "Epoch-16 validation Pareto surfaces — descriptive, no selected winner"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _cost_frontier_plot(rows, fronts, output):
    colors = {
        "greedy": "#2369bd",
        "self_seed_greedy": "#d55e00",
        "dbscan": "#009e73",
        "hdbscan": "#8b5cf6",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.5))
    for ax, (key, label) in zip(axes, COST_METRICS.items()):
        rate = key != "candidates_per_event"
        scale = 100 if rate else 1
        for clusterer, color in colors.items():
            selected = [
                row for row in rows
                if row["clusterer"] == clusterer
                and row["n_matched_gt10"] > 0
                and row["promotable"]
            ]
            ax.scatter(
                [scale * row[key] for row in selected],
                [100 * row["idea_def2"] for row in selected],
                s=12, alpha=0.28, color=color, label=clusterer,
            )
        front_rows = [row for row in rows if row["point"] in fronts[key]]
        front_rows.sort(key=lambda row: row[key])
        ax.plot(
            [scale * row[key] for row in front_rows],
            [100 * row["idea_def2"] for row in front_rows],
            "k.--", linewidth=1.0, markersize=5, label="Pareto",
        )
        ax.set_xlabel(label)
        ax.set_ylabel("IDEA definition 2 [%]")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Candidate-cost Pareto surfaces — descriptive, no selected winner"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _oracle_plot(rows, output):
    oracle = [row for row in rows if not row["promotable"]]
    if not oracle:
        return False
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    scatter = ax.scatter(
        [100 * row["ggtf10_unassigned_per_matched"] for row in oracle],
        [100 * row["idea_def2"] for row in oracle],
        c=[row["td"] for row in oracle],
        s=42,
        marker="x",
        cmap="viridis",
    )
    ax.set_xlabel("GGTF >10 unassigned / matched [%]")
    ax.set_ylabel("IDEA definition 2 [%]")
    ax.set_title("Non-promotable truth-position helix appendix")
    ax.grid(alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="base distance threshold")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return True


def _signature(
    stage, clusterer, tbeta, td, min_hits, merge_td=0.0,
    attach_td=0.0, helix_tol=0.0,
):
    return (
        stage, clusterer, float(tbeta), float(td), int(min_hits),
        float(merge_td), float(attach_td), float(helix_tol),
    )


def _validate_campaign(root: Path, rows: list[dict]) -> dict:
    campaign = json.loads((root / "campaign_manifest.json").read_text())
    campaign_sha = campaign.pop("campaign_sha256")
    if _canonical_hash(campaign) != campaign_sha:
        raise RuntimeError("campaign manifest self-hash is invalid")
    package = Path(__file__).resolve().parents[2]
    if _source_hashes(package) != campaign["source_sha256"]:
        raise RuntimeError("current evaluator/report source differs from campaign")
    for identity in campaign["inputs"].values():
        path = Path(identity["path"])
        if _sha256(path) != identity["sha256"]:
            raise RuntimeError(f"campaign input content changed: {path}")
    for marker_name in ("SWEEP_COMPLETE", "ORACLE_APPENDIX_COMPLETE"):
        marker = json.loads((root / marker_name).read_text())
        if marker.get("campaign_sha256") != campaign_sha:
            raise RuntimeError(f"{marker_name} does not match campaign")
    if any(row["sweep_provenance_sha256"] != campaign_sha for row in rows):
        raise RuntimeError("one or more point summaries have mixed provenance")

    expected = {
        _signature("full_grid", clusterer, tbeta, td, 4)
        for clusterer in ("greedy", "self_seed_greedy")
        for tbeta in TBETAS for td in TDS
    }
    expected.update({
        _signature("density", clusterer, 0.5, td, 4)
        for clusterer in ("dbscan", "hdbscan") for td in TDS
    })
    for source in json.loads((root / "refinement_sources.json").read_text()):
        clusterer, tbeta, td = (
            source["clusterer"], source["tbeta"], source["td"]
        )
        for min_hits, merge, attach in (
            (3, 0.0, 0.0), (5, 0.0, 0.0),
            (4, 0.5 * td, 0.0), (4, td, 0.0),
            (4, 0.0, 0.5 * td), (4, 0.0, td),
            (4, 0.5 * td, 0.5 * td),
        ):
            expected.add(_signature(
                "refinement", clusterer, tbeta, td, min_hits,
                merge, attach,
            ))
    for source in json.loads(
        (root / "density_refinement_sources.json").read_text()
    ):
        for min_hits in (3, 5):
            expected.add(_signature(
                "density_refinement", source["clusterer"], 0.5,
                source["td"], min_hits,
            ))
    oracle = json.loads((root / "oracle_helix/README.json").read_text())
    if oracle.get("promotable") is not False:
        raise RuntimeError("oracle appendix is not marked non-promotable")
    for source in oracle["sources"]:
        for tolerance in oracle["tolerances"]:
            expected.add(_signature(
                "oracle_helix", source["clusterer"], source["tbeta"],
                source["td"], 4, helix_tol=tolerance,
            ))
    actual = {
        _signature(
            row["stage"], row["clusterer"], row["tbeta"], row["td"],
            row["min_cluster_hits"], row["merge_td"], row["attach_td"],
            row["helix_tol"],
        )
        for row in rows
    }
    if actual != expected or len(rows) != len(actual):
        raise RuntimeError(
            f"sweep configuration mismatch: expected={len(expected)}, "
            f"actual_unique={len(actual)}, rows={len(rows)}, "
            f"missing={len(expected - actual)}, extra={len(actual - expected)}"
        )
    return {**campaign, "campaign_sha256": campaign_sha}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.root.glob("*/*/*/summary.json"))
    if not paths:
        raise SystemExit(f"no sweep summaries below {args.root}")
    rows = [_flatten(args.root, path) for path in paths]
    campaign = _validate_campaign(args.root, rows)

    fronts = {key: _pareto(rows, key) for key in Y_METRICS}
    cost_fronts = {
        key: _pareto_pair(rows, key, "idea_def2")
        for key in COST_METRICS
    }
    for key, points in fronts.items():
        for row in rows:
            row[f"pareto_{key}"] = row["point"] in points
    for key, points in cost_fronts.items():
        for row in rows:
            row[f"pareto_idea_def2_vs_{key}"] = row["point"] in points

    frame = pl.DataFrame(rows).sort(
        ["stage", "clusterer", "tbeta", "td", "min_cluster_hits"]
    )
    frame.write_csv(args.root / "all_points.csv")
    pareto_union = set().union(*fronts.values(), *cost_fronts.values())
    frame.filter(pl.col("point").is_in(pareto_union)).write_csv(
        args.root / "pareto_frontier.csv"
    )

    for clusterer in ("greedy", "self_seed_greedy"):
        for key, label in (
            ("idea_def2", "IDEA definition 2"),
            ("low_pt_01_02_def2", "low-pT 0.1–0.2 GeV definition 2"),
            ("ggtf10_unassigned_per_matched", "GGTF >10 unassigned / matched"),
        ):
            _heatmap(
                rows, clusterer, key, label,
                args.root / f"heatmap_{clusterer}_{key}.png",
            )
    _frontier_plot(rows, fronts, args.root / "pareto_frontiers.png")
    _cost_frontier_plot(
        rows, cost_fronts, args.root / "candidate_cost_frontiers.png"
    )
    wrote_oracle = _oracle_plot(
        rows, args.root / "oracle_helix_appendix.png"
    )

    by_algorithm = {}
    for clusterer in sorted({row["clusterer"] for row in rows}):
        selected = [
            row for row in rows
            if row["clusterer"] == clusterer and row["promotable"]
        ]
        valid_selected = [
            row for row in selected if row["n_matched_gt10"] > 0
        ]
        by_algorithm[clusterer] = {
            "points": len(selected),
            "nonpromotable_oracle_points": sum(
                row["clusterer"] == clusterer and not row["promotable"]
                for row in rows
            ),
            "idea_def2_range": [
                min(row["idea_def2"] for row in valid_selected),
                max(row["idea_def2"] for row in valid_selected),
            ],
            "low_pt_01_02_def2_range": [
                min(row["low_pt_01_02_def2"] for row in valid_selected),
                max(row["low_pt_01_02_def2"] for row in valid_selected),
            ],
            "ggtf10_range": [
                min(row["ggtf10_unassigned_per_matched"] for row in valid_selected),
                max(row["ggtf10_unassigned_per_matched"] for row in valid_selected),
            ],
        }
    report = {
        "scope": (
            "epoch-16 immutable no-keep-all validation cache; no keep-all or "
            "Loopers input is configured in the campaign"
        ),
        "campaign_sha256": campaign["campaign_sha256"],
        "campaign_inputs": campaign["inputs"],
        "campaign_source_sha256": campaign["source_sha256"],
        "exact_configuration_set_verified": True,
        "decision_policy": (
            "Pareto frontier only. No operating-point winner selected and no "
            "official selector changed."
        ),
        "adaptive_selection_caveat": (
            "Refinement sources were selected on these same 500 validation "
            "events. Even/odd deltas are stability diagnostics, not independent "
            "confirmatory inference; any promoted point requires frozen "
            "evaluation on independent keep-all and Loopers datasets."
        ),
        "metric_definitions": {
            "IDEA_definition_1": "best-cluster purity > 0.75",
            "IDEA_definition_2": (
                "best-cluster purity > 0.5 and hit efficiency > 0.5"
            ),
            "GGTF_gt10": (
                "Hungarian-unassigned / assigned for candidates with >10 hits"
            ),
            "candidate_fake_fraction": "unassigned / all candidates",
            "all_target_no_cuts": (
                "all cached primary targets before min-3 truth relabelling"
            ),
            "min3_target_no_reconstruction_cuts": (
                "targets surviving GGTF create_garbage_label(minNumHits=3)"
            ),
        },
        "algorithm_parameter_semantics": {
            "greedy": "tbeta=seed threshold, td=assignment radius",
            "self_seed_greedy": (
                "same thresholds, but seeds already claimed by a higher-beta "
                "seed cannot spawn another cluster"
            ),
            "dbscan": "tbeta unused, td=eps, min_samples=min_cluster_hits",
            "hdbscan": (
                "tbeta unused, td=cluster_selection_epsilon, "
                "min_samples=min_cluster_hits"
            ),
        },
        "points": len(rows),
        "degenerate_points_without_matched_gt10": sum(
            row["n_matched_gt10"] == 0 for row in rows
        ),
        "nonpromotable_oracle_points": sum(
            not row["promotable"] for row in rows
        ),
        "pareto_union_points": len(pareto_union),
        "pareto_counts": {key: len(value) for key, value in fronts.items()},
        "candidate_cost_pareto_counts": {
            key: len(value) for key, value in cost_fronts.items()
        },
        "algorithms": by_algorithm,
        "max_even_odd_deltas": {
            key: max(
                row[key] for row in rows if row[key] is not None
            )
            for key in (
                "split_delta_idea_def2",
                "split_delta_idea_def1",
                "split_delta_idea_hit_eff",
                "split_delta_low_pt_01_02_def1",
                "split_delta_low_pt_01_02_def2",
                "split_delta_low_pt_02_05_def1",
                "split_delta_low_pt_02_05_def2",
                "split_delta_ggtf10",
            )
        },
        "undefined_even_odd_ggtf_deltas": sum(
            row["split_delta_ggtf10"] is None for row in rows
        ),
        "artifacts": {
            "all_points": str((args.root / "all_points.csv").resolve()),
            "pareto": str((args.root / "pareto_frontier.csv").resolve()),
            "frontier_plot": str((args.root / "pareto_frontiers.png").resolve()),
            "candidate_cost_plot": str(
                (args.root / "candidate_cost_frontiers.png").resolve()
            ),
            **({
                "oracle_plot": str(
                    (args.root / "oracle_helix_appendix.png").resolve()
                )
            } if wrote_oracle else {}),
        },
    }
    (args.root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
