"""Plot the three principal GGTF IDEA validation slices for one fixed OP.

The left panel overlays the two published matching definitions: De Vita slide
24 uses purity-only definition 1, while CHEP Figure 4 uses definition 2. The
middle and right panels reproduce the other CHEP Figure 4 observables.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.eval.plot_fcc_metrics import binom_ci, save_fig
from src.eval.plotstyle import COL, plt, yticks
from src.eval.report_final_slices import nearest_truth_distance


def _binary_bins(
    frame: pl.DataFrame, column: str, edges: np.ndarray, value: str,
    geometric: bool,
) -> list[dict]:
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = frame.filter(
            (pl.col(column) >= float(low)) & (pl.col(column) < float(high))
        )
        n = len(selected)
        if n == 0:
            continue
        passed = int(selected[value].cast(pl.Int64).sum())
        rate, error_low, error_high = binom_ci(passed, n)
        center = np.sqrt(low * high) if geometric else 0.5 * (low + high)
        rows.append({
            "low": float(low),
            "high": float(high),
            "center": float(center),
            "n": n,
            "value": float(rate),
            "error_low": float(error_low),
            "error_high": float(error_high),
        })
    return rows


def _mean_bins(
    frame: pl.DataFrame, column: str, edges: np.ndarray, value: str,
    geometric: bool,
) -> list[dict]:
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = frame.filter(
            (pl.col(column) >= float(low)) & (pl.col(column) < float(high))
        )
        values = selected[value].drop_nulls().to_numpy()
        if len(values) == 0:
            continue
        center = np.sqrt(low * high) if geometric else 0.5 * (low + high)
        error = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        rows.append({
            "low": float(low),
            "high": float(high),
            "center": float(center),
            "n": len(values),
            "value": float(np.mean(values)),
            "error_low": error,
            "error_high": error,
        })
    return rows


def _draw(ax, rows: list[dict], label: str, color: str, marker: str) -> None:
    x = np.asarray([row["center"] for row in rows])
    y = np.asarray([row["value"] for row in rows])
    low = np.asarray([row["error_low"] for row in rows])
    high = np.asarray([row["error_high"] for row in rows])
    ax.errorbar(
        x, y, yerr=[low, high], fmt=f"{marker}-", color=color, ecolor=color,
        linewidth=1.5, markersize=4.2, label=label,
    )


def _finish_efficiency_axis(ax, xlabel: str, log_x: bool = False) -> None:
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylim(0.45, 1.02)
    ax.axhline(1.0, color=COL["ref"], linestyle=":", linewidth=0.8)
    yticks(ax, 0.1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", default="epoch-16 preliminary")
    args = parser.parse_args()

    source = Path(args.cache_dir) / "cache.parquet"
    candidate_summary_path = Path(args.cache_dir) / "summary.json"
    tracks = pl.read_parquet(source)
    candidate_summary = json.loads(candidate_summary_path.read_text())
    required = {
        "seed", "event_id", "pt", "eta", "phi", "purity_of_match",
        "efficiency_per_hit", "is_reconstructable_idea",
    }
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"cache is missing columns {sorted(missing)}")
    idea = (
        tracks.filter(pl.col("is_reconstructable_idea"))
        .with_columns(
            (pl.col("purity_of_match") > 0.75).alias("definition_1"),
            (
                (pl.col("purity_of_match") > 0.5)
                & (pl.col("efficiency_per_hit") > 0.5)
            ).alias("definition_2"),
        )
    )
    idea = idea.with_columns(
        pl.Series("nearest_truth_delta_eta_phi", nearest_truth_distance(idea))
    )

    # Match GGTF's local plotting helper: exp-spaced pT bins, step 0.2.
    pt_edges = np.exp(np.arange(np.log(0.1), np.log(60.0) + 0.2, 0.2))
    proximity_edges = np.asarray([0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    curves = {
        "definition_1_vs_pt": _binary_bins(
            idea, "pt", pt_edges, "definition_1", geometric=True
        ),
        "definition_2_vs_pt": _binary_bins(
            idea, "pt", pt_edges, "definition_2", geometric=True
        ),
        "mean_hit_efficiency_vs_pt": _mean_bins(
            idea, "pt", pt_edges, "efficiency_per_hit", geometric=True
        ),
        "definition_2_vs_proximity": _binary_bins(
            idea.drop_nulls("nearest_truth_delta_eta_phi"),
            "nearest_truth_delta_eta_phi", proximity_edges,
            "definition_2", geometric=False,
        ),
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.2))
    _draw(
        axes[0], curves["definition_1_vs_pt"],
        "definition 1: purity >75%", COL["standard"], "o",
    )
    _draw(
        axes[0], curves["definition_2_vs_pt"],
        "definition 2: purity >50% and hit eff. >50%", COL["strict"], "s",
    )
    _finish_efficiency_axis(axes[0], r"$p_\mathrm{T}$ [GeV]", log_x=True)
    axes[0].set_ylabel("Tracking efficiency")
    axes[0].set_title("IDEA matching efficiency")
    axes[0].legend(loc="lower right", fontsize=7.5)

    _draw(
        axes[1], curves["mean_hit_efficiency_vs_pt"],
        "mean shared-hit fraction", COL["eff"], "o",
    )
    _finish_efficiency_axis(axes[1], r"$p_\mathrm{T}$ [GeV]", log_x=True)
    axes[1].set_ylabel("Mean hit efficiency")
    axes[1].set_title("IDEA hit efficiency")
    axes[1].legend(loc="lower right", fontsize=8)

    _draw(
        axes[2], curves["definition_2_vs_proximity"],
        "definition 2", COL["strict"], "o",
    )
    _finish_efficiency_axis(
        axes[2], r"Nearest truth-track $\Delta(\eta,\phi)$", log_x=True
    )
    axes[2].set_ylabel("Tracking efficiency")
    axes[2].set_title("IDEA efficiency vs proximity")
    axes[2].legend(loc="lower right", fontsize=8)

    # Describe the operating point actually used, read from the point's own
    # summary. Asserting "fixed training operating point" misreports any point
    # that came out of a sweep rather than the training defaults.
    operating_point = (
        f"{candidate_summary.get('clusterer', 'unknown')}"
        f" tbeta={candidate_summary.get('tbeta')}"
        f" td={candidate_summary.get('td')}"
        f" min_hits={candidate_summary.get('min_cluster_hits')}"
    )
    if candidate_summary.get("attach_td"):
        operating_point += f" attach={candidate_summary['attach_td']}"
    if candidate_summary.get("merge_td"):
        operating_point += f" merge={candidate_summary['merge_td']}"
    fig.suptitle(
        f"{args.tag} · no-keep-all validation · {operating_point}",
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, str(output / "ggtf_important_panels.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    _draw(
        ax, curves["definition_1_vs_pt"],
        "slide 24 / definition 1", COL["standard"], "o",
    )
    _draw(
        ax, curves["definition_2_vs_pt"],
        "CHEP Fig. 4 / definition 2", COL["strict"], "s",
    )
    _finish_efficiency_axis(ax, r"$p_\mathrm{T}$ [GeV]", log_x=True)
    ax.set_ylabel("Tracking efficiency")
    ax.set_title(f"{args.tag}: IDEA efficiency vs $p_T$")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, str(output / "eff_vs_pt_ggtf_definitions.png"))
    plt.close(fig)

    flat_rows = []
    for metric, rows in curves.items():
        flat_rows.extend({"metric": metric, **row} for row in rows)
    with (output / "ggtf_important_bins.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    summary = {
        "tag": args.tag,
        "source": str(source.resolve()),
        "n_reconstructable_idea": len(idea),
        "definition_1": float(idea["definition_1"].mean()),
        "definition_2": float(idea["definition_2"].mean()),
        "mean_hit_efficiency": float(idea["efficiency_per_hit"].mean()),
        "ggtf_style_candidate_size_gt10": {
            "unassigned_per_matched": candidate_summary.get(
                "ggtf_fake_rate_gt10"
            ),
            "candidate_fake_fraction": candidate_summary.get(
                "ggtf_candidate_fake_fraction_gt10"
            ),
            "n_candidates": candidate_summary.get("ggtf_n_cand_gt10"),
            "n_matched": candidate_summary.get("ggtf_n_matched_gt10"),
            "n_fake_clone": candidate_summary.get("ggtf_n_fake_clone_gt10"),
            "n_fake_spurious": candidate_summary.get("ggtf_n_fake_spurious_gt10"),
        },
        "definitions": {
            "definition_1": "best-cluster hit purity > 75%",
            "definition_2": "best-cluster purity > 50% and hit efficiency > 50%",
            "denominator": (
                "CGA IDEA reconstructable: pT>0.1 GeV, n_hits_total>10, "
                "15<theta<165 deg, charged, gen_status in {0,1}"
            ),
            "proximity": (
                "nearest other IDEA-reconstructable truth particle in wrapped eta-phi"
            ),
        },
        "caveat": (
            "Preliminary fixed-OP validation; not the post-training milestone "
            "operating-point selection and not a digitized GGTF overlay."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
