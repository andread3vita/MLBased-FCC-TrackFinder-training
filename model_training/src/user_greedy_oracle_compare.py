"""Oracle-merge headroom for the user-greedy clusterer alongside reference.

For each inference setup (reference + user-greedy at 5 OPs) reads:
  - the unmerged FCC cache (deployment metric)
  - oracle-merge caches at T ∈ {0.50, 0.65, 0.75}

and produces a comparison table + per-pT plots showing the gap between
the deployment-realistic number and the oracle upper bound at each
strict-T threshold.

This is the same diagnostic as Section 7 of overview.md, extended to
the user-greedy OPs (incl. colleagues' tbeta=0.60, td=0.30).

Outputs:
  <out_dir>/oracle_compare_user.md
  <out_dir>/oracle_compare_user_strict50.png
  <out_dir>/oracle_compare_user_strict99.png
  <out_dir>/oracle_compare_user_dashboard.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


PT_BINS = [
    (0.10, 0.15), (0.15, 0.20), (0.20, 0.30), (0.30, 0.50),
    (0.50, 1.00), (1.00, 3.00), (3.00, 30.0),
]

INFERENCE_SETUPS = [
    ("v35 greedy (tbeta=0.10, td=0.20)",   "v35_fcc_unlimited"),
    ("v35 greedy (tbeta=0.10, td=0.25)",   "v35_fcc_td0.25"),
    ("v35 HDBSCAN (mcs=5)",                "v35_fcc_hdbscan"),
    ("v35 user-greedy (0.10, 0.05)",       "v35_fcc_user_tbeta0.1_td0.05"),
    ("v35 user-greedy (0.10, 0.20)",       "v35_fcc_user_tbeta0.1_td0.2"),
    ("v35 user-greedy (0.10, 0.25)",       "v35_fcc_user_tbeta0.1_td0.25"),
    ("v35 user-greedy (0.60, 0.30)",       "v35_fcc_user_tbeta0.6_td0.3"),
    ("v35 user-greedy (0.60, 0.40)",       "v35_fcc_user_tbeta0.6_td0.4"),
    ("v35 user-greedy (0.60, 0.70)",       "v35_fcc_user_tbeta0.6_td0.7"),
    ("v35 user-greedy (0.60, 1.00)",       "v35_fcc_user_tbeta0.6_td1.0"),
    ("v35 user-greedy (0.70, 0.05)",       "v35_fcc_user_tbeta0.7_td0.05"),
]


def cache_suffix(base_dir: str, T: float | None) -> str:
    if T is None:
        return base_dir
    if base_dir.startswith("v35_fcc_user"):
        return f"{base_dir}_merge_T{T}"
    return f"{base_dir}_merge_T{T:.2f}"


def overall(cache: str, T_strict_list: list[float]) -> dict | None:
    p = Path("eval_results") / cache
    if not (p / "cache.parquet").exists():
        return None
    df = pl.read_parquet(p / "cache.parquet").filter(
        pl.col("is_reconstructable_idea")
    )
    cl = pl.read_parquet(p / "cache_clusters.parquet")
    n = max(df.height, 1)
    matched = df.filter(pl.col("matched"))
    out = {"n_idea": df.height, "n_clusters": cl.height,
           "loose": matched.height / n}
    for T in T_strict_list:
        out[f"strict{int(T*100)}"] = (
            matched.filter(pl.col("efficiency_per_hit") >= T).height / n
        )
    return out


def per_pt(cache: str, T_strict: float | None) -> list[tuple[float, float]] | None:
    """Per-pT match rate. T_strict=None gives loose match (purity>=0.75 only)."""
    p = Path("eval_results") / cache
    if not (p / "cache.parquet").exists():
        return None
    df = pl.read_parquet(p / "cache.parquet").filter(
        pl.col("is_reconstructable_idea")
    )
    rows = []
    for lo, hi in PT_BINS:
        sl = df.filter((pl.col("pt") >= lo) & (pl.col("pt") < hi))
        if sl.height == 0:
            rows.append((0.5 * (lo + hi), float("nan")))
            continue
        matched = sl.filter(pl.col("matched"))
        if T_strict is None:
            v = matched.height / sl.height
        else:
            v = (
                matched.filter(pl.col("efficiency_per_hit") >= T_strict).height
                / sl.height
            )
        rows.append((0.5 * (lo + hi), v))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="eval_results/v35_user_oracle_compare")
    ap.add_argument(
        "--strict_thresholds", nargs="+", type=float,
        default=[0.50, 0.75, 0.90, 0.99],
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    md = [
        "# User-greedy oracle-merge headroom (v35, IDEA cut)\n",
        "",
        "**Diagnostic**, not deployment: oracle rows use ground-truth "
        "`mc_idx` at inference to merge predicted clusters that share a "
        "dominant truth particle (purity \u2265 T_assoc). The gap "
        "vs the unmerged row is the inference-time merging headroom.\n",
    ]
    for inf_label, base in INFERENCE_SETUPS:
        md.append(f"\n## {inf_label}\n")
        md.append("| variant | n_clusters | loose | strict50 | strict75 | strict90 | strict99 |")
        md.append("|---|---:|---:|---:|---:|---:|---:|")
        variants = [
            ("unmerged (deployment)", None),
            ("oracle T=0.50", 0.50),
            ("oracle T=0.65", 0.65),
            ("oracle T=0.75", 0.75),
        ]
        for var_label, T in variants:
            cache_name = cache_suffix(base, T)
            ov = overall(cache_name, args.strict_thresholds)
            if ov is None:
                continue
            md.append(
                f"| {var_label} | {ov['n_clusters']} | "
                f"{ov['loose']*100:.2f}% | {ov['strict50']*100:.2f}% | "
                f"{ov['strict75']*100:.2f}% | {ov['strict90']*100:.2f}% | "
                f"{ov['strict99']*100:.2f}% |"
            )
    (out / "oracle_compare_user.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {out / 'oracle_compare_user.md'}")

    cmap = plt.colormaps["tab10"]
    n_setups = len(INFERENCE_SETUPS)
    ncols = 4
    nrows = (n_setups + ncols - 1) // ncols

    for T in [0.50, 0.99]:
        T_int = int(T * 100)
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows),
                                 sharey=True)
        axes_flat = axes.flat
        variants = [
            ("unmerged (deployment)", None, "-"),
            ("oracle T=0.50", 0.50, ":"),
            ("oracle T=0.65", 0.65, "--"),
            ("oracle T=0.75", 0.75, "-."),
        ]
        for ax_idx, (inf_label, base) in enumerate(INFERENCE_SETUPS):
            ax = axes_flat[ax_idx]
            for k, (var_label, Tval, ls) in enumerate(variants):
                cache_name = cache_suffix(base, Tval)
                rows = per_pt(cache_name, T)
                if rows is None:
                    continue
                xs = [r[0] for r in rows]
                ys = [r[1] for r in rows]
                ax.plot(xs, ys, marker="o", color=cmap(k),
                        linestyle=ls, label=var_label,
                        linewidth=1.4, markersize=4)
            ax.set_xscale("log")
            ax.set_xlabel("pT [GeV]")
            ax.set_ylim(0.0, 1.02)
            ax.grid(alpha=0.3)
            ax.set_title(inf_label, fontsize=10)
            if ax_idx % ncols == 0:
                ax.set_ylabel(f"strict{T_int} match")
            ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
        for j in range(n_setups, nrows * ncols):
            axes_flat[j].axis("off")
        fig.suptitle(
            f"strict{T_int} match: deployment vs oracle merge (IDEA). "
            "Solid = deployment (real); dashed = oracle (truth-cheating).",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / f"oracle_compare_user_strict{T_int}.png", dpi=130)
        plt.close(fig)
        print(f"Wrote {out / f'oracle_compare_user_strict{T_int}.png'}")

    metric_rows = [
        (None, "loose"),
        (0.50, "strict50"),
        (0.99, "strict99"),
    ]
    fig, axes = plt.subplots(
        len(metric_rows), n_setups,
        figsize=(3.4 * n_setups, 3.4 * len(metric_rows)),
        sharey="row",
    )
    variants = [
        ("unmerged",   None, "-"),
        ("T=0.50",     0.50, ":"),
        ("T=0.65",     0.65, "--"),
        ("T=0.75",     0.75, "-."),
    ]
    for col, (inf_label, base) in enumerate(INFERENCE_SETUPS):
        for row, (T, metric_label) in enumerate(metric_rows):
            ax = axes[row][col]
            for k, (var_label, Tval, ls) in enumerate(variants):
                cache_name = cache_suffix(base, Tval)
                pts = per_pt(cache_name, T)
                if pts is None:
                    continue
                xs = [r[0] for r in pts]
                ys = [r[1] for r in pts]
                ax.plot(xs, ys, marker="o", color=cmap(k),
                        linestyle=ls, label=var_label,
                        linewidth=1.2, markersize=3)
            ax.set_xscale("log")
            ax.set_ylim(0.0, 1.02)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(inf_label.replace("user ", ""), fontsize=8)
            if row == len(metric_rows) - 1:
                ax.set_xlabel("pT [GeV]", fontsize=8)
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=9)
            ax.legend(loc="lower right", fontsize=5.5, framealpha=0.9)
    fig.suptitle(
        "Oracle-merge headroom (v35, IDEA). "
        "Rows: loose / strict50 / strict99. "
        "Solid = deployment; dotted/dashed = oracle merge at T_assoc.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "oracle_compare_user_dashboard.png", dpi=130)
    plt.close(fig)
    print(f"Wrote {out / 'oracle_compare_user_dashboard.png'}")


if __name__ == "__main__":
    main()
