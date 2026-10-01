"""Overlay fixed-operating-point FCC efficiency curves across checkpoints."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.eval.sample_provenance import describe_samples

from src.eval.plot_fcc_metrics import (
    _setup_eff_axes,
    binned_log_efficiency,
    save_fig,
)
from src.eval.plotstyle import COL


PT_EDGES = np.logspace(np.log10(0.1), np.log10(30.0), 31)
COLORS = [COL["grey"], COL["keepall"], COL["standard"]]
# Beyond three curves the palette above repeats and the overlay becomes
# unreadable, so switch to a qualitative map and vary the line style too.
WIDE_COLORS = list(plt.get_cmap("tab10").colors)
LINESTYLES = ("-", "--", "-.", ":")


def _styles(count: int):
    if count <= len(COLORS):
        return [(COLORS[i], "-") for i in range(count)]
    return [
        (WIDE_COLORS[i % len(WIDE_COLORS)], LINESTYLES[i // len(WIDE_COLORS)])
        for i in range(count)
    ]


def _curve(df: pl.DataFrame, generator_only: bool):
    selected = df.filter(pl.col("is_reconstructable_idea"))
    if generator_only:
        selected = selected.filter(pl.col("gen_status") == 1)
    selected = selected.with_columns(
        (pl.col("purity_of_match") > 0.75).alias("definition_1")
    )
    return binned_log_efficiency(
        selected, "pt", PT_EDGES, value_col="definition_1"
    ), selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", action="append", required=True, metavar="LABEL=PATH",
        help="label and cache.parquet path; repeat for each checkpoint",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--title",
        default="CIRCE checkpoint progression at fixed self-seed operating point",
    )
    parser.add_argument(
        "--subtitle",
        default="tbeta=0.1, td=0.2, min cluster hits=4",
        help="Caption text placed after the sample provenance, which is read "
             "from the campaign manifest rather than passed in. Override when "
             "the overlaid curves do not share one operating point, otherwise "
             "the caption misreports them.",
    )
    parser.add_argument("--filename", default="eff_vs_pt_checkpoint_progression")
    args = parser.parse_args()

    inputs = []
    for item in args.cache:
        label, separator, path = item.partition("=")
        if not separator:
            parser.error(f"--cache must be LABEL=PATH, got {item!r}")
        inputs.append((label, Path(path)))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"definition": "best-cluster purity > 75%", "checkpoints": {}}

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), sharey=True)
    styles = _styles(len(inputs))
    for index, (label, path) in enumerate(inputs):
        df = pl.read_parquet(path)
        color, linestyle = styles[index]
        for ax, generator_only in zip(axes, (False, True)):
            (centers, values, err_lo, err_hi, _), selected = _curve(
                df, generator_only
            )
            ax.errorbar(
                centers, values, yerr=[err_lo, err_hi], marker="o",
                linestyle=linestyle,
                color=color, ecolor=color, markersize=3.5, linewidth=1.35,
                label=label,
            )
            key = "generator_only" if generator_only else "all_idea"
            summary["checkpoints"].setdefault(label, {})[key] = {
                "n": len(selected),
                "definition_1": float(selected["definition_1"].mean()),
                "source": str(path.resolve()),
            }

    titles = (
        "All IDEA targets (includes detector secondaries)",
        r"Generator particles only ($\mathrm{genStatus}=1$)",
    )
    for ax, title in zip(axes, titles):
        ax.set_xscale("log")
        ax.set_xlabel(r"$p_\mathrm{T}$ [GeV]")
        ax.set_title(title)
        _setup_eff_axes(ax, ymin=0.5)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("Tracking efficiency (definition 1)")
    # Provenance is read from the caches themselves, so a caption cannot claim a
    # sample the figure does not show (M86).
    caption = f"Source: {describe_samples(path for _, path in inputs)}"
    if args.subtitle:
        caption += f" · {args.subtitle}"
    fig.suptitle(args.title, fontsize=11)
    fig.text(0.5, -0.01, caption, ha="center", fontsize=7)
    fig.tight_layout()
    save_fig(fig, str(output / f"{args.filename}.png"))
    plt.close(fig)

    summary["title"] = args.title
    summary["subtitle"] = caption
    # Named after the figure rather than the script, so an overlay of operating
    # points does not leave a file called "checkpoint_progression" beside it.
    (output / f"{args.filename}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
