"""FCC-style plots for v36-EF.

Reads the per-track + per-cluster Parquet caches written by
`eval_fcc_metrics_v36.py` and produces slide-style efficiency / fake-rate
plots with binomial CI error bars.

Plots produced (slide references in parentheses):
  - eff_vs_pt_idea.png         (slide 24, IDEA cuts)
  - eff_vs_pt_cld.png          (slide 16, CLD cuts)
  - eff_vs_theta.png           (efficiency vs theta in degrees, IDEA cuts)
  - eff_vs_eta.png             (efficiency vs pseudorapidity, IDEA cuts)
  - eff_vs_vertexR.png         (slide 26, displaced cuts; R = sqrt(vx^2+vy^2))
  - eff_vs_nhits.png           (efficiency vs detector hits per particle)
  - nhit_distribution.png      (per-particle hit count, log y)
  - fake_rate_summary.png      (overall + per-pT fake rate)
  - metrics_table.md           (overall purity / efficiency / fake-rate table)

Usage:
    cd model_training
    PYTHONPATH=. python src/plot_fcc_metrics_v36.py \
        --cache_dir eval_results/v36ef_fcc \
        --output_dir eval_results/v36ef_fcc/plots \
        --tag v36-EF
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import polars as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import binomtest
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def binom_ci(matched: int, total: int, conf: float = 0.68):
    """68% binomial Clopper-Pearson CI. Falls back to Wald if scipy missing."""
    if total <= 0:
        return 0.0, 0.0, 0.0
    p = matched / total
    if HAVE_SCIPY:
        ci = binomtest(int(matched), int(total)).proportion_ci(conf)
        return p, p - ci.low, ci.high - p
    z = 1.0  # ~68% Wald approx
    se = np.sqrt(max(p * (1 - p), 0.0) / total)
    return p, z * se, z * se


def binned_efficiency(df: pl.DataFrame, col: str, edges,
                       value_col: str = "matched"):
    """Compute per-bin efficiency + 68% binomial CI errors and bin centers.

    Returns: (centers, p, err_lo, err_hi, ns).
    """
    centers, ps, e_lo, e_hi, ns = [], [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = df.filter((pl.col(col) >= lo) & (pl.col(col) < hi))
        n = len(sub)
        if n == 0:
            continue
        m = int(sub[value_col].cast(pl.Int64).sum())
        p, lo_err, hi_err = binom_ci(m, n)
        centers.append(0.5 * (lo + hi))
        ps.append(p)
        e_lo.append(lo_err)
        e_hi.append(hi_err)
        ns.append(n)
    return (np.array(centers), np.array(ps),
            np.array(e_lo), np.array(e_hi), np.array(ns))


def binned_log_efficiency(df: pl.DataFrame, col: str, edges,
                           value_col: str = "matched"):
    """Same as binned_efficiency but uses geometric mean as bin center."""
    centers, ps, e_lo, e_hi, ns = [], [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = df.filter((pl.col(col) >= lo) & (pl.col(col) < hi))
        n = len(sub)
        if n == 0:
            continue
        m = int(sub[value_col].cast(pl.Int64).sum())
        p, lo_err, hi_err = binom_ci(m, n)
        centers.append(np.sqrt(lo * hi))
        ps.append(p)
        e_lo.append(lo_err)
        e_hi.append(hi_err)
        ns.append(n)
    return (np.array(centers), np.array(ps),
            np.array(e_lo), np.array(e_hi), np.array(ns))


def _setup_eff_axes(ax, ymin=0.0):
    ax.set_ylim(ymin, 1.05)
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)


def plot_eff_vs_pt(df: pl.DataFrame, mask_col: str, edges,
                   out_path: str, title: str, log_x: bool = True):
    sub = df.filter(pl.col(mask_col))
    if len(sub) == 0:
        print(f"  [skip] {out_path}: no rows after mask {mask_col}")
        return
    centers, p, e_lo, e_hi, ns = binned_log_efficiency(
        sub, "pt", edges, value_col="matched"
    )
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(centers, p, yerr=[e_lo, e_hi], fmt="o-", color="C0",
                ecolor="C0", capsize=3, markersize=5, linewidth=1.2,
                label=f"v36-EF ({mask_col.replace('is_reconstructable_', '')} cuts)")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(r"$p_T$ [GeV]")
    ax.set_ylabel("Tracking efficiency (75% purity match)")
    ax.set_title(title)
    _setup_eff_axes(ax, ymin=0.5)
    ax.legend(loc="lower right")
    if len(centers) <= 15:
        for x, y, n in zip(centers, p, ns):
            ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=7,
                        color="0.4")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_eff_vs_var(df: pl.DataFrame, mask_col: str, col: str, edges,
                    xlabel: str, out_path: str, title: str,
                    ymin: float = 0.5):
    sub = df.filter(pl.col(mask_col))
    if len(sub) == 0:
        print(f"  [skip] {out_path}: no rows after mask {mask_col}")
        return
    centers, p, e_lo, e_hi, ns = binned_efficiency(
        sub, col, edges, value_col="matched"
    )
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(centers, p, yerr=[e_lo, e_hi], fmt="o-", color="C0",
                ecolor="C0", capsize=3, markersize=5, linewidth=1.2,
                label=f"v36-EF ({mask_col.replace('is_reconstructable_', '')} cuts)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Tracking efficiency (75% purity match)")
    ax.set_title(title)
    _setup_eff_axes(ax, ymin=ymin)
    ax.legend(loc="lower right")
    for x, y, n in zip(centers, p, ns):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=7,
                    color="0.4")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_nhit_distribution(df: pl.DataFrame, out_path: str, title: str):
    """Per-particle n_hits distribution, separately for IDEA-reconstructable
    and the whole population."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    nh_all = df["n_hits_total"].to_numpy()
    nh_idea = df.filter(pl.col("is_reconstructable_idea"))["n_hits_total"].to_numpy()
    bins = np.linspace(0, max(nh_all.max(), 200), 80)
    ax.hist(nh_all, bins=bins, alpha=0.55, color="C7",
            label=f"all primaries (N={len(nh_all)})")
    ax.hist(nh_idea, bins=bins, alpha=0.85, color="C0",
            label=f"IDEA-reconstructable (N={len(nh_idea)})")
    ax.axvline(10, color="red", linestyle="--", linewidth=1.0,
               label="N_hits > 10 cut")
    ax.set_yscale("log")
    ax.set_xlabel("Detector hits per primary particle (vtx + dc)")
    ax.set_ylabel("Particles per bin")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_fake_rate_summary(cluster_df: pl.DataFrame, particle_df: pl.DataFrame,
                            out_path: str, title: str):
    """Two panels:
       (left) overall fake rate (bar chart for IDEA / CLD / "no cuts")
       (right) fake rate vs reconstructed cluster size
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    n_total = len(cluster_df)
    n_fake_idea = int(cluster_df["is_fake_idea"].sum())
    n_fake_cld = int(cluster_df["is_fake_cld"].sum())
    n_low_pur = int(cluster_df.filter(pl.col("purity") < 0.75).height)

    fr_idea, idea_lo, idea_hi = binom_ci(n_fake_idea, n_total)
    fr_cld, cld_lo, cld_hi = binom_ci(n_fake_cld, n_total)
    fr_pur, pur_lo, pur_hi = binom_ci(n_low_pur, n_total)

    labels = ["fake (IDEA)", "fake (CLD)", "low-purity\n(<75%)"]
    values = [fr_idea, fr_cld, fr_pur]
    errs = [[idea_lo, cld_lo, pur_lo], [idea_hi, cld_hi, pur_hi]]
    colors = ["C0", "C2", "C7"]
    bars = axes[0].bar(labels, values, yerr=errs, capsize=4,
                       color=colors, alpha=0.85, edgecolor="black",
                       linewidth=0.6)
    for bar, v in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                     f"{v * 100:.1f}%", ha="center", fontsize=10)
    axes[0].set_ylabel("Fraction of reconstructed clusters")
    axes[0].set_title(f"Cluster classification ({n_total:,} clusters total)")
    axes[0].set_ylim(0, max(max(values) * 1.4, 0.15))
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].axhline(0.08, color="red", linestyle="--", linewidth=1.0,
                    label="FCC slide 24 (fake = 8%)")
    axes[0].legend(loc="upper right", fontsize=9)

    size_edges = [2, 3, 5, 8, 12, 20, 35, 60, 100, 200, 500, 2000]
    centers, p, e_lo, e_hi, ns = binned_efficiency(
        cluster_df, "cluster_size", size_edges, value_col="is_fake_idea"
    )
    axes[1].errorbar(centers, p, yerr=[e_lo, e_hi], fmt="o-", color="C0",
                     capsize=3, markersize=5, label="fake (IDEA reco set)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Cluster size [hits]")
    axes[1].set_ylabel("Fake fraction")
    axes[1].set_title("Fake rate vs cluster size (IDEA cuts)")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.3, which="both")
    axes[1].axhline(0.08, color="red", linestyle="--", linewidth=1.0)
    for x, y, n in zip(centers, p, ns):
        axes[1].annotate(f"n={n}", (x, y), textcoords="offset points",
                         xytext=(0, -14), ha="center", fontsize=7,
                         color="0.4")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def write_metrics_table(particle_df: pl.DataFrame, cluster_df: pl.DataFrame,
                        out_path: str, tag: str):
    def stats(df, mask_col=None):
        if mask_col is not None:
            df = df.filter(pl.col(mask_col))
        n = len(df)
        if n == 0:
            return None
        n_match = int(df["matched"].cast(pl.Int64).sum())
        eff_p, eff_lo, eff_hi = binom_ci(n_match, n)
        eff_per_hit = float(df["efficiency_per_hit"].mean())
        return {"n": n, "match_rate": eff_p, "match_lo": eff_lo,
                "match_hi": eff_hi, "eff_per_hit": eff_per_hit}

    n_clu = len(cluster_df)
    cluster_purity = float(cluster_df["purity"].mean())
    n_fake_idea = int(cluster_df["is_fake_idea"].sum())
    n_fake_cld = int(cluster_df["is_fake_cld"].sum())
    fr_idea, fr_idea_lo, fr_idea_hi = binom_ci(n_fake_idea, n_clu)
    fr_cld, fr_cld_lo, fr_cld_hi = binom_ci(n_fake_cld, n_clu)

    rows = [
        ("no cuts",   stats(particle_df, None)),
        ("IDEA",      stats(particle_df, "is_reconstructable_idea")),
        ("CLD",       stats(particle_df, "is_reconstructable_cld")),
        ("displaced", stats(particle_df, "is_reconstructable_displaced")),
    ]

    md = []
    md.append(f"# v36-EF FCC-style metrics — {tag}\n")
    md.append("Operating point: greedy `tbeta=0.025`, `td=0.10` (v36 phase-2 sweep optimum).\n")
    md.append(f"Total reconstructed clusters: **{n_clu:,}**.  "
              f"Mean cluster purity: **{cluster_purity:.3f}**.\n")
    md.append("")
    md.append("## Tracking efficiency (per particle, ≥75% purity match)\n")
    md.append("| Reconstructable cut | N particles | Efficiency (per hit) | Match rate (75% pur.) |")
    md.append("|---|---:|---:|---:|")
    for name, s in rows:
        if s is None:
            md.append(f"| {name} | 0 | — | — |")
        else:
            md.append(
                f"| {name} | {s['n']:,} | {s['eff_per_hit']:.3f} | "
                f"**{s['match_rate']:.3f}** "
                f"(+{s['match_hi']:.3f}/-{s['match_lo']:.3f}, 68% CI) |"
            )
    md.append("")
    md.append("## Fake rate (per cluster)\n")
    md.append("| Reco set | Fake rate | 68% CI |")
    md.append("|---|---:|---|")
    md.append(f"| IDEA  | **{fr_idea:.3f}** | "
              f"+{fr_idea_hi:.3f}/-{fr_idea_lo:.3f} |")
    md.append(f"| CLD   | **{fr_cld:.3f}** | "
              f"+{fr_cld_hi:.3f}/-{fr_cld_lo:.3f} |")
    md.append("")
    md.append("## Direct comparison vs FCC slide 24\n")
    md.append("| Metric | FCC (slide 24, no track fit) | v36-EF (this run) |")
    md.append("|---|---:|---:|")
    md.append(f"| Fake rate (IDEA) | 8.0% | **{fr_idea * 100:.1f}%** |")
    fcc_eff = "—"
    md.append(f"| Reco efficiency (IDEA) | {fcc_eff} | **{rows[1][1]['match_rate'] * 100:.1f}%** |")
    md.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(md))
    print(f"  saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="FCC-style plots for v36-EF")
    parser.add_argument("--cache_dir", type=str,
                        default="eval_results/v36ef_fcc")
    parser.add_argument("--output_dir", type=str,
                        default="eval_results/v36ef_fcc/plots")
    parser.add_argument("--tag", type=str, default="v36-EF")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    particle_path = os.path.join(args.cache_dir, "cache.parquet")
    cluster_path = os.path.join(args.cache_dir, "cache_clusters.parquet")
    if not os.path.exists(particle_path):
        raise FileNotFoundError(particle_path)
    if not os.path.exists(cluster_path):
        raise FileNotFoundError(cluster_path)

    particle_df = pl.read_parquet(particle_path)
    cluster_df = pl.read_parquet(cluster_path)
    print(f"Loaded {len(particle_df)} particle rows from {particle_path}")
    print(f"Loaded {len(cluster_df)} cluster rows from {cluster_path}")

    # binning — IDEA edges match FCC slide 24 (~30 log-spaced bins, 0.1-30 GeV)
    pt_edges_idea = list(np.logspace(np.log10(0.1), np.log10(30.0), 31))
    pt_edges_cld = list(np.logspace(np.log10(0.1), np.log10(50.0), 25))
    theta_edges = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
    eta_edges = list(np.linspace(-3.0, 3.0, 13))
    vertexR_edges = [0.0, 5.0, 20.0, 50.0, 100.0, 200.0, 400.0, 800.0,
                     1500.0, 3000.0]
    nhit_edges = [4, 7, 10, 15, 20, 30, 50, 80, 130, 200, 350, 600]

    print("\n=== Plotting ===")
    plot_eff_vs_pt(
        particle_df, "is_reconstructable_idea", pt_edges_idea,
        os.path.join(args.output_dir, "eff_vs_pt_idea.png"),
        title=f"{args.tag}: tracking efficiency vs $p_T$ (IDEA cuts, slide-24 style)",
    )
    plot_eff_vs_pt(
        particle_df, "is_reconstructable_cld", pt_edges_cld,
        os.path.join(args.output_dir, "eff_vs_pt_cld.png"),
        title=f"{args.tag}: tracking efficiency vs $p_T$ (CLD cuts, slide-16 style)",
    )
    plot_eff_vs_var(
        particle_df, "is_reconstructable_idea", "theta_deg", theta_edges,
        xlabel=r"$\theta$ [degrees]",
        out_path=os.path.join(args.output_dir, "eff_vs_theta.png"),
        title=f"{args.tag}: tracking efficiency vs $\\theta$ (IDEA cuts)",
    )
    plot_eff_vs_var(
        particle_df, "is_reconstructable_idea", "eta", eta_edges,
        xlabel=r"$\eta$",
        out_path=os.path.join(args.output_dir, "eff_vs_eta.png"),
        title=f"{args.tag}: tracking efficiency vs $\\eta$ (IDEA cuts)",
    )
    plot_eff_vs_var(
        particle_df, "is_reconstructable_displaced", "vertex_r", vertexR_edges,
        xlabel=r"Vertex $R = \sqrt{v_x^2 + v_y^2}$ [mm]",
        out_path=os.path.join(args.output_dir, "eff_vs_vertexR.png"),
        title=f"{args.tag}: tracking efficiency vs vertex R (slide-26 style, displaced cuts)",
        ymin=0.0,
    )
    plot_eff_vs_var(
        particle_df.with_columns(pl.col("n_hits_total").cast(pl.Float64)),
        "is_reconstructable_idea", "n_hits_total", nhit_edges,
        xlabel="Detector hits per particle (vtx + dc)",
        out_path=os.path.join(args.output_dir, "eff_vs_nhits.png"),
        title=f"{args.tag}: tracking efficiency vs N_hits (IDEA cuts)",
    )
    plot_nhit_distribution(
        particle_df,
        os.path.join(args.output_dir, "nhit_distribution.png"),
        title=f"{args.tag}: N_hits distribution per primary particle",
    )
    plot_fake_rate_summary(
        cluster_df, particle_df,
        os.path.join(args.output_dir, "fake_rate_summary.png"),
        title=f"{args.tag}: fake-rate breakdown",
    )
    write_metrics_table(
        particle_df, cluster_df,
        os.path.join(args.output_dir, "metrics_table.md"),
        tag=args.tag,
    )

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
