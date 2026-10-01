"""Efficiency vs pT with the per-bin target population overlaid.

An efficiency curve alone hides how many targets each point is averaged over. On
the Loopers sample in particular the low-pT bins hold very few targets, so the
leftmost points carry wide statistical uncertainty and must not be read with the
same confidence as the high-pT points. This plots both on shared pT axes: the
efficiency with binomial errors on the left axis, and the number of
IDEA-reconstructable targets per bin as bars on the right axis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.eval.plot_fcc_metrics import (
    _setup_eff_axes,
    binned_log_efficiency,
    save_fig,
)

# Same bins as analyze_pt_adaptive_ceiling.py so the numbers line up.
PT_EDGES = np.array([0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.3, 2.0, 3.5, 6.0, 30.0])


def _prepare(path: Path, generator_only: bool) -> pl.DataFrame:
    df = pl.read_parquet(path).filter(pl.col("is_reconstructable_idea"))
    if generator_only:
        df = df.filter(pl.col("gen_status") == 1)
    return df.with_columns(
        (pl.col("purity_of_match") > 0.75).alias("def1"),
        (
            (pl.col("purity_of_match") > 0.5)
            & (pl.col("efficiency_per_hit") > 0.5)
        ).alias("def2"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", action="append", required=True, metavar="LABEL=PATH",
        help="dataset label and its cache.parquet; repeat for each panel",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--filename", default="eff_vs_pt_with_population")
    parser.add_argument(
        "--generator-only", action="store_true",
        help="restrict to generator particles (gen_status == 1), the M74 cut",
    )
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = []
    for item in args.cache:
        label, separator, path = item.partition("=")
        if not separator:
            parser.error(f"--cache must be LABEL=PATH, got {item!r}")
        inputs.append((label, Path(path)))

    fig, axes = plt.subplots(
        1, len(inputs), figsize=(5.8 * len(inputs), 4.3), sharey=True,
        squeeze=False,
    )
    summary: dict[str, dict] = {}

    for index, (label, path) in enumerate(inputs):
        ax = axes[0][index]
        df = _prepare(path, args.generator_only)

        centers, def1, lo1, hi1, counts = binned_log_efficiency(
            df, "pt", PT_EDGES, value_col="def1"
        )
        _, def2, lo2, hi2, _ = binned_log_efficiency(
            df, "pt", PT_EDGES, value_col="def2"
        )

        # Population steps first so the curves draw on top. Use stairs against
        # the true bin edges: bars of constant linear width do not tile bins on
        # a log axis. Counts are recomputed per edge pair because
        # binned_log_efficiency drops empty bins and would misalign.
        population = ax.twinx()
        edge_counts = np.array([
            len(df.filter((pl.col("pt") >= lo) & (pl.col("pt") < hi)))
            for lo, hi in zip(PT_EDGES[:-1], PT_EDGES[1:])
        ])
        population.stairs(
            np.maximum(edge_counts, 1), PT_EDGES, fill=True,
            color="0.86", edgecolor="0.62", linewidth=0.6, zorder=1,
        )
        population.set_yscale("log")
        population.set_ylabel(
            "IDEA targets per bin (shaded, log scale)", fontsize=9
        )
        population.set_ylim(1, max(edge_counts.max(), 1) * 6)
        for lo, hi, count in zip(PT_EDGES[:-1], PT_EDGES[1:], edge_counts):
            if count == 0:
                continue
            population.annotate(
                f"{count:,}", (np.sqrt(lo * hi), count),
                textcoords="offset points", xytext=(0, 3), ha="center",
                fontsize=6.5, color="0.35",
            )

        ax.set_zorder(population.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.errorbar(
            centers, def1, yerr=[lo1, hi1], marker="o", linestyle="-",
            color="#1f77b4", ecolor="#1f77b4", markersize=4,
            linewidth=1.5, label="definition 1: purity > 75%", zorder=3,
        )
        ax.errorbar(
            centers, def2, yerr=[lo2, hi2], marker="s", linestyle="-",
            color="#2ca02c", ecolor="#2ca02c", markersize=4,
            linewidth=1.5,
            label="definition 2: purity > 50% and hit eff > 50%", zorder=3,
        )
        ax.set_xscale("log")
        ax.set_xlabel(r"true $p_\mathrm{T}$ [GeV]")
        ax.set_title(label, fontsize=10)
        _setup_eff_axes(ax, ymin=0.4)
        ax.grid(alpha=0.25, zorder=0)
        ax.legend(loc="lower right", fontsize=7.5)

        summary[label] = {
            "source": str(path.resolve()),
            "n_targets": len(df),
            "population_per_bin": edge_counts.tolist(),
            "bins": [
                {
                    "pt_center": float(c),
                    "n": int(n),
                    "def1": float(d1),
                    "def2": float(d2),
                }
                for c, n, d1, d2 in zip(centers, counts, def1, def2)
            ],
        }

    axes[0][0].set_ylabel("Tracking efficiency")
    scope = (
        "generator particles only (genStatus = 1)" if args.generator_only
        else "all IDEA-reconstructable targets"
    )
    fig.suptitle(
        args.title
        or f"Efficiency vs true $p_T$ with per-bin target population · {scope}",
        fontsize=11,
    )
    fig.text(
        0.5, -0.03,
        "Bars show how many targets each efficiency point averages over. "
        "pT is the TRUTH pT from the MC record, not a reconstructed quantity. "
        "Frozen reference operating point.",
        ha="center", fontsize=7,
    )
    fig.tight_layout()
    save_fig(fig, str(args.output_dir / f"{args.filename}.png"))
    plt.close(fig)

    (args.output_dir / f"{args.filename}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    for label, info in summary.items():
        print(f"--- {label}  ({info['n_targets']:,} targets)")
        for entry in info["bins"]:
            print(
                f"   pT~{entry['pt_center']:>6.2f}  n={entry['n']:>5}  "
                f"def1={entry['def1'] * 100:>6.2f}%  "
                f"def2={entry['def2'] * 100:>6.2f}%"
            )


if __name__ == "__main__":
    main()
