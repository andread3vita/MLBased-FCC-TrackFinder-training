"""Measure the ceiling of pT-dependent clustering parameters.

Choosing the clustering radius per pT bin cannot be done at inference time: pT is
a truth quantity, and a candidate's pT is not known until it is reconstructed. So
the per-bin best is an ORACLE and is not promotable, in the same sense as the
truth-position helix appendix.

It is still worth measuring, because it bounds what any legitimate pT-adaptive
scheme (a cascade over decreasing radii, or a curvature-based proxy) could
possibly buy. If the oracle envelope is barely above the best single fixed point,
there is no reason to build the real thing.

Reports the envelope under def1 and def2 separately, because def1 is purity-only
and rewards fragmentation: a scheme that looks strong under def1 while losing hit
efficiency has not reconstructed anything better.
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

from src.eval.sample_provenance import describe_samples

# Slide-style logarithmic pT bins, coarse enough for stable per-bin counts.
PT_EDGES = np.array([0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.3, 2.0, 3.5, 6.0, 30.0])
METRICS = ("def1", "def2")


def _prepare(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path).filter(pl.col("is_reconstructable_idea"))
    return df.with_columns(
        (pl.col("purity_of_match") > 0.75).alias("def1"),
        (
            (pl.col("purity_of_match") > 0.5)
            & (pl.col("efficiency_per_hit") > 0.5)
        ).alias("def2"),
    )


def _per_bin(df: pl.DataFrame) -> list[dict]:
    rows = []
    for low, high in zip(PT_EDGES[:-1], PT_EDGES[1:]):
        sel = df.filter((pl.col("pt") >= low) & (pl.col("pt") < high))
        if len(sel) == 0:
            rows.append({"low": float(low), "high": float(high), "n": 0})
            continue
        rows.append({
            "low": float(low),
            "high": float(high),
            "n": len(sel),
            "def1": float(sel["def1"].mean()),
            "def2": float(sel["def2"].mean()),
            "hit_eff": float(sel["efficiency_per_hit"].mean()),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", action="append", required=True, metavar="LABEL=PATH",
        help="operating point label and its cache.parquet; repeat",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    points: dict[str, list[dict]] = {}
    overall: dict[str, dict] = {}
    cache_paths: list[Path] = []
    for item in args.cache:
        label, separator, path = item.partition("=")
        if not separator:
            parser.error(f"--cache must be LABEL=PATH, got {item!r}")
        cache_paths.append(Path(path))
        df = _prepare(Path(path))
        points[label] = _per_bin(df)
        overall[label] = {
            "def1": float(df["def1"].mean()),
            "def2": float(df["def2"].mean()),
            "hit_eff": float(df["efficiency_per_hit"].mean()),
            "n": len(df),
        }

    labels = list(points)
    n_bins = len(PT_EDGES) - 1
    report = {
        "status": "ORACLE, NOT PROMOTABLE",
        "why": (
            "Selecting the clustering radius per pT bin requires the truth pT of "
            "a candidate before it is reconstructed. This bounds any legitimate "
            "pT-adaptive scheme; it is not itself a usable operating point."
        ),
        "pt_edges": PT_EDGES.tolist(),
        "operating_points": overall,
        "bins": [],
        "envelope": {},
    }

    # Per-bin winners and the envelope they imply, weighted by bin population.
    for metric in METRICS:
        weighted = 0.0
        total = 0
        best_single = max(labels, key=lambda label: overall[label][metric])
        for index in range(n_bins):
            counts = [points[label][index].get("n", 0) for label in labels]
            if not any(counts):
                continue
            n = counts[0]
            winner = max(
                labels, key=lambda label: points[label][index].get(metric, 0.0)
            )
            weighted += points[winner][index][metric] * n
            total += n
        envelope = weighted / total if total else float("nan")
        report["envelope"][metric] = {
            "oracle_envelope": envelope,
            "best_single_point": best_single,
            "best_single_value": overall[best_single][metric],
            "gain_over_best_single": envelope - overall[best_single][metric],
        }

    for index in range(n_bins):
        entry = {
            "pt_low": float(PT_EDGES[index]),
            "pt_high": float(PT_EDGES[index + 1]),
            "n": points[labels[0]][index].get("n", 0),
            "by_point": {
                label: {
                    metric: points[label][index].get(metric)
                    for metric in (*METRICS, "hit_eff")
                }
                for label in labels
            },
        }
        for metric in METRICS:
            values = {
                label: points[label][index].get(metric)
                for label in labels
                if points[label][index].get(metric) is not None
            }
            if values:
                winner = max(values, key=lambda label: values[label])
                entry[f"{metric}_winner"] = winner
                entry[f"{metric}_best"] = values[winner]
        report["bins"].append(entry)

    (args.output_dir / "pt_adaptive_ceiling.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    # Plot: every fixed point plus the per-bin oracle envelope, for both metrics.
    centers = np.sqrt(PT_EDGES[:-1] * PT_EDGES[1:])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharex=True)
    colors = plt.get_cmap("tab10").colors
    for ax, metric in zip(axes, METRICS):
        for order, label in enumerate(labels):
            values = [points[label][i].get(metric, np.nan) for i in range(n_bins)]
            ax.plot(
                centers, values, marker="o", markersize=3.5, linewidth=1.2,
                color=colors[order % len(colors)], label=label, alpha=0.85,
            )
        envelope = [
            max(
                points[label][i].get(metric, float("-inf")) for label in labels
            )
            for i in range(n_bins)
        ]
        ax.plot(
            centers, envelope, color="black", linewidth=2.0, linestyle="--",
            marker="s", markersize=4, label="per-bin oracle envelope",
        )
        ax.set_xscale("log")
        ax.set_xlabel(r"$p_\mathrm{T}$ [GeV]")
        ax.set_ylabel(f"Tracking efficiency ({metric})")
        ax.set_title(
            "Definition 1 (purity > 0.75 only)" if metric == "def1"
            else "Definition 2 (purity > 0.5 and hit eff > 0.5)"
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(
        "Ceiling of pT-dependent clustering radius (oracle, not promotable)",
        fontsize=11,
    )
    fig.text(
        0.5, -0.02,
        f"Source: {describe_samples(cache_paths)} · envelope selects the best "
        "operating point per truth-pT bin, which requires truth and is not "
        "achievable at inference",
        ha="center", fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(
        args.output_dir / "pt_adaptive_ceiling.png", dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("=== per-bin winners ===")
    print(
        f"{'pT bin':<16}{'n':>7}"
        + "".join(f"{label[:15]:>17}" for label in labels)
        + f"{'def2 winner':>30}"
    )
    for entry in report["bins"]:
        if not entry["n"]:
            continue
        cells = "".join(
            f"{(entry['by_point'][label]['def2'] or 0) * 100:>17.2f}"
            for label in labels
        )
        print(
            f"{entry['pt_low']:>6.2f}-{entry['pt_high']:<9.2f}{entry['n']:>7d}"
            f"{cells}{entry.get('def2_winner', '-'):>30}"
        )
    print()
    for metric in METRICS:
        info = report["envelope"][metric]
        print(
            f"{metric}: oracle envelope {info['oracle_envelope'] * 100:.2f}% vs "
            f"best single point {info['best_single_value'] * 100:.2f}% "
            f"({info['best_single_point']}), "
            f"gain {info['gain_over_best_single'] * 100:+.2f} points"
        )


if __name__ == "__main__":
    main()
