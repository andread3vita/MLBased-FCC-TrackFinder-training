"""Score one cache under every denominator GGTF's own notebooks actually use.

`trackingEfficiencyPlot` (plotting_tools.py:325) takes its cuts as arguments and
defaults to `applyConstraints=False`, which applies only a theta window. Their
notebooks then call it with three different configurations. There is therefore no
single "GGTF denominator", and the low-pT gap against their slides cannot be
interpreted without knowing which one a given figure used.

This scores the same predictions under all of them, so the spread attributable to
denominator choice alone is visible and separable from any model difference.

Their cuts, transcribed from the call sites (M81):

  trackFinder_training, ndf_raw ............ genStatus [1],    nHits > 3,  theta 15-165
  trackFinder_training, extraAssignation ... genStatus [0, 1], nHits > 10, theta 15-165
  invariance_fix, ndf_raw .................. genStatus [0, 1], nHits > 5,  theta 15-175
  applyConstraints=False (library default) . theta 10-170 only

Note their hit cut is `numSIhits + numCDChits > minNumHits`, strictly greater,
and their pT cut is `pT > 0`. Ours additionally requires charge != 0 and
pT > 0.1 GeV, which is kept as a separate row rather than folded in.
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

from src.eval.sample_provenance import describe_sample

# GGTF's own binning: exp-spaced, 0.1 to 60 GeV, log step 0.2.
PT_EDGES = np.exp(np.arange(np.log(0.1), np.log(60.0) + 0.2, 0.2))

# name -> (gen_status set or None, min_hits_exclusive, theta_lo, theta_hi,
#          require_charged, min_pt)
#
# The named rows are the configurations that appear verbatim in their notebooks,
# plus ours. These are the ones plotted. The systematic grid below then crosses
# every axis so no combination is left unexamined.
NAMED_VARIANTS: dict[str, tuple] = {
    "ggtf_generator_only_nhits3": ((1,), 3, 15.0, 165.0, False, 0.0),
    "ggtf_allstatus_nhits10": ((0, 1), 10, 15.0, 165.0, False, 0.0),
    "ggtf_allstatus_nhits5_theta175": ((0, 1), 5, 15.0, 175.0, False, 0.0),
    "ggtf_library_default_theta_only": (None, 0, 10.0, 170.0, False, 0.0),
    "ours_idea_reconstructable": ((0, 1), 10, 15.0, 165.0, True, 0.1),
    "ours_but_generator_only": ((1,), 10, 15.0, 165.0, True, 0.1),
}

GRID_GEN_STATUS = {"gen1": (1,), "gen01": (0, 1)}
GRID_NHITS = (3, 5, 10)
GRID_THETA = {
    "theta15-165": (15.0, 165.0),
    "theta15-175": (15.0, 175.0),
    "theta10-170": (10.0, 170.0),
}
# Whether our two extra cuts (charge != 0, pT > 0.1) are layered on top.
GRID_EXTRA = {"ggtfcuts": (False, 0.0), "ourcuts": (True, 0.1)}


def _full_grid() -> dict[str, tuple]:
    grid: dict[str, tuple] = {}
    for gs_name, gs in GRID_GEN_STATUS.items():
        for nhits in GRID_NHITS:
            for theta_name, (lo, hi) in GRID_THETA.items():
                for extra_name, (charged, min_pt) in GRID_EXTRA.items():
                    key = f"{gs_name}_nhits{nhits}_{theta_name}_{extra_name}"
                    grid[key] = (gs, nhits, lo, hi, charged, min_pt)
    return grid


def _apply(df: pl.DataFrame, spec: tuple) -> pl.DataFrame:
    gen_status, min_hits, theta_lo, theta_hi, charged, min_pt = spec
    theta_deg = pl.col("theta").degrees()
    out = df.filter((theta_deg > theta_lo) & (theta_deg < theta_hi))
    if gen_status is not None:
        out = out.filter(pl.col("gen_status").is_in(list(gen_status)))
    if min_hits:
        out = out.filter(pl.col("n_hits_total") > min_hits)
    if charged:
        out = out.filter(pl.col("charge") != 0)
    if min_pt:
        out = out.filter(pl.col("pt") > min_pt)
    return out


def _binned(df: pl.DataFrame, column: str):
    centers, values, counts = [], [], []
    for lo, hi in zip(PT_EDGES[:-1], PT_EDGES[1:]):
        sel = df.filter((pl.col("pt") >= lo) & (pl.col("pt") < hi))
        if len(sel) == 0:
            continue
        centers.append(float(np.sqrt(lo * hi)))
        values.append(float(sel[column].mean()))
        counts.append(len(sel))
    return np.array(centers), np.array(values), np.array(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", default="CGA epoch 24")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(args.cache).with_columns(
        (pl.col("purity_of_match") > 0.75).alias("def1"),
        (
            (pl.col("purity_of_match") > 0.5)
            & (pl.col("efficiency_per_hit") > 0.5)
        ).alias("def2"),
    )

    report = {
        "scope": (
            "Same predictions, same events, same operating point. Only the "
            "denominator changes. Differences here are definitional, not model "
            "quality."
        ),
        "source": str(args.cache.resolve()),
        "pt_edges": PT_EDGES.tolist(),
        "variants": {},
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), sharey=True)
    colors = plt.get_cmap("tab10").colors
    low_pt_rows = []

    def score(spec: tuple) -> dict | None:
        sub = _apply(df, spec)
        if len(sub) == 0:
            return None
        gen_status, min_hits, theta_lo, theta_hi, charged, min_pt = spec
        low = sub.filter((pl.col("pt") >= 0.1) & (pl.col("pt") < 0.2))
        return {
            "cuts": {
                "gen_status": list(gen_status) if gen_status else "any",
                "n_hits_total_exclusive_min": min_hits,
                "theta_deg": [theta_lo, theta_hi],
                "require_charged": charged,
                "min_pt_gev": min_pt,
            },
            "n_targets": len(sub),
            "def1": float(sub["def1"].mean()),
            "def2": float(sub["def2"].mean()),
            "low_pt_0p1_0p2": {
                "n": len(low),
                "def1": float(low["def1"].mean()) if len(low) else None,
                "def2": float(low["def2"].mean()) if len(low) else None,
            },
        }

    for order, (name, spec) in enumerate(NAMED_VARIANTS.items()):
        entry = score(spec)
        if entry is None:
            continue
        report["variants"][name] = entry
        low_pt_rows.append((name, entry))

        sub = _apply(df, spec)
        for ax, column in zip(axes, ("def1", "def2")):
            centers, values, _ = _binned(sub, column)
            ax.plot(
                centers, values, marker="o", markersize=3.2, linewidth=1.4,
                color=colors[order % len(colors)], label=name, alpha=0.9,
            )

    # Exhaustive grid: every genStatus x nHits x theta x extra-cut combination.
    grid_rows = []
    for name, spec in _full_grid().items():
        entry = score(spec)
        if entry is None:
            continue
        report.setdefault("full_grid", {})[name] = entry
        grid_rows.append((name, entry))
    def fmt(value) -> str:
        return "" if value is None else f"{value:.6f}"

    grid_csv = ["variant,gen_status,n_hits_min,theta_lo,theta_hi,charged,"
                "min_pt,n_targets,def1,def2,low_pt_n,low_pt_def1,low_pt_def2"]
    for name, entry in grid_rows:
        cuts = entry["cuts"]
        low = entry["low_pt_0p1_0p2"]
        grid_csv.append(",".join([
            name,
            str(cuts["gen_status"]).replace(",", "+").replace(" ", ""),
            str(cuts["n_hits_total_exclusive_min"]),
            str(cuts["theta_deg"][0]),
            str(cuts["theta_deg"][1]),
            str(cuts["require_charged"]),
            str(cuts["min_pt_gev"]),
            str(entry["n_targets"]),
            fmt(entry["def1"]),
            fmt(entry["def2"]),
            str(low["n"]),
            fmt(low["def1"]),
            fmt(low["def2"]),
        ]))
    (args.output_dir / "ggtf_denominator_grid.csv").write_text(
        "\n".join(grid_csv) + "\n"
    )

    for ax, column, title in zip(
        axes, ("def1", "def2"),
        ("Definition 1 (purity > 75%), as on slide 24",
         "Definition 2 (purity > 50% and hit eff > 50%), as in CHEP Fig 4"),
    ):
        ax.set_xscale("log")
        ax.set_xlabel(r"true $p_\mathrm{T}$ [GeV]")
        ax.set_ylabel(f"Tracking efficiency ({column})")
        ax.set_title(title, fontsize=9.5)
        ax.set_ylim(0.4, 1.02)
        ax.grid(alpha=0.25)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
        ax.legend(fontsize=6.8, loc="lower right")
    fig.suptitle(
        f"{args.tag}: one model, every GGTF denominator their notebooks use",
        fontsize=11,
    )
    fig.text(
        0.5, -0.03,
        f"Source: {describe_sample(args.cache)}, frozen reference operating "
        "point. GGTF binning (exp, 0.1-60 GeV, step 0.2). Spread between "
        "curves is denominator choice alone.",
        ha="center", fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(
        args.output_dir / "eff_vs_pt_ggtf_denominators.png", dpi=200,
        bbox_inches="tight",
    )
    fig.savefig(
        args.output_dir / "eff_vs_pt_ggtf_denominators.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    (args.output_dir / "ggtf_denominator_variants.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print(f"{'variant':<36}{'n':>8}{'def1':>9}{'def2':>9}"
          f"{'lowpT n':>9}{'lowpT def1':>12}")
    for name, entry in low_pt_rows:
        low = entry["low_pt_0p1_0p2"]
        low_def1 = (
            f"{low['def1'] * 100:.2f}%" if low["def1"] is not None else "--"
        )
        print(
            f"{name:<36}{entry['n_targets']:>8}"
            f"{entry['def1'] * 100:>8.2f}%{entry['def2'] * 100:>8.2f}%"
            f"{low['n']:>9}{low_def1:>12}"
        )


if __name__ == "__main__":
    main()
