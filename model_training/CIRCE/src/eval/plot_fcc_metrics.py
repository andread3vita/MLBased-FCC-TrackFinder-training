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
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import polars as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.eval.plotstyle import COL, yticks  # applies the shared theme on import

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
    yticks(ax, 0.1)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)


def save_fig(fig, out_path: str):
    """Save the figure as the given .png (print dpi) plus a vector .pdf sibling."""
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    root, _ = os.path.splitext(out_path)
    fig.savefig(root + ".pdf", bbox_inches="tight")


def plot_eff_vs_pt(df: pl.DataFrame, mask_col: str, edges,
                   out_path: str, title: str, log_x: bool = True):
    sub = df.filter(pl.col(mask_col))
    if len(sub) == 0:
        print(f"  [skip] {out_path}: no rows after mask {mask_col}")
        return
    centers, p, e_lo, e_hi, ns = binned_log_efficiency(
        sub, "pt", edges, value_col="matched"
    )
    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    ax.errorbar(centers, p, yerr=[e_lo, e_hi], fmt="o-", color=COL["keepall"],
                ecolor=COL["keepall"], markersize=4.5, linewidth=1.6,
                label="CIRCE")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(r"$p_\mathrm{T}$ [GeV]")
    ax.set_ylabel("Track match rate ($>75\\%$ purity)")
    ax.set_title(title)
    _setup_eff_axes(ax, ymin=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, out_path)
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
    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    ax.errorbar(centers, p, yerr=[e_lo, e_hi], fmt="o-", color=COL["keepall"],
                ecolor=COL["keepall"], markersize=4.5, linewidth=1.6,
                label="CIRCE")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Track match rate ($>75\\%$ purity)")
    ax.set_title(title)
    _setup_eff_axes(ax, ymin=ymin)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_nhit_distribution(df: pl.DataFrame, out_path: str, title: str):
    """Per-particle n_hits distribution, separately for IDEA-reconstructable
    and the whole population."""
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    nh_all = df["n_hits_total"].to_numpy()
    nh_idea = df.filter(pl.col("is_reconstructable_idea"))["n_hits_total"].to_numpy()
    # log-x: the full truth population (dominated by few-hit secondaries in the
    # keep-all sample) and the O(100)-hit reconstructable bulk are orders of
    # magnitude apart; a linear axis hides one or the other.
    lo = max(1, int(nh_all[nh_all > 0].min())) if (nh_all > 0).any() else 1
    bins = np.logspace(np.log10(lo), np.log10(max(nh_all.max(), 200)), 60)
    ax.hist(nh_all, bins=bins, alpha=0.55, color=COL["grey"],
            label=f"all truth particles (N={len(nh_all):,})")
    ax.hist(nh_idea, bins=bins, alpha=0.85, color=COL["standard"],
            label=f"IDEA-reconstructable (N={len(nh_idea):,})")
    ax.axvline(10, color=COL["keepall"], linestyle="--", linewidth=1.2,
               label="$N_\\mathrm{hits} > 10$ cut")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Detector hits per particle (vtx + dc)")
    ax.set_ylabel("Particles per bin")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_fake_rate_summary(cluster_df: pl.DataFrame, particle_df: pl.DataFrame,
                            out_path: str, title: str):
    """Single panel: IDEA fake fraction vs reconstructed cluster size.

    The overall IDEA fake rate is quoted in the text/table, not as a bar here;
    this figure shows *where* the fakes live — small clusters — which is the
    interesting structure (fakes are mostly track fragments).
    """
    fig, ax = plt.subplots(figsize=(5.6, 4.0))

    size_edges = [1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200, 500, 2000]
    centers, p, e_lo, e_hi, ns = binned_efficiency(
        cluster_df, "cluster_size", size_edges, value_col="is_fake_idea"
    )
    h_pts = ax.errorbar(
        centers, p, yerr=[e_lo, e_hi], fmt="o-", color=COL["fake"],
        ecolor=COL["fake"], markersize=5, linewidth=1.6,
        label="Legacy IDEA-selection fake fraction",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Reconstructed cluster size [hits]")
    ax.set_ylabel("Fake fraction")
    ax.set_title(title)
    # zoom to where the data live (fake fraction stays below ~20%)
    ax.set_ylim(0, 0.25); yticks(ax, 0.05)
    # This candidate-normalized legacy selection metric is not normalized like
    # GGTF's unassigned/matched ratio, so do not overlay the published ~8%.
    overall = float(cluster_df["is_fake_idea"].mean())
    h_avg = ax.axhline(overall, color=COL["ink"], linestyle="-", linewidth=1.6,
                       label=f"CIRCE average ({100 * overall:.1f}%)")
    ax.legend(
        [h_pts, h_avg],
        ["Legacy IDEA-selection fake fraction",
         f"CIRCE average ({100 * overall:.1f}%)"],
        loc="upper right",
    )

    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def write_metrics_table(particle_df: pl.DataFrame, cluster_df: pl.DataFrame,
                        out_path: str, tag: str, tbeta=None, td=None,
                        clusterer=None):
    def stats(df, mask_col=None):
        if mask_col is not None:
            df = df.filter(pl.col(mask_col))
        n = len(df)
        if n == 0:
            return None
        n_match = int((df["purity_of_match"] > 0.75).sum())
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
    ggtf_gt10 = cluster_df.filter(pl.col("cluster_size") > 10)
    n_ggtf_gt10 = len(ggtf_gt10)
    n_ggtf_match_gt10 = int(ggtf_gt10["ggtf_assigned"].sum())
    n_ggtf_fake_gt10 = int((~ggtf_gt10["ggtf_assigned"]).sum())
    candidate_fr, candidate_lo, candidate_hi = binom_ci(
        n_ggtf_fake_gt10, n_ggtf_gt10
    )
    ggtf_ratio = (
        n_ggtf_fake_gt10 / n_ggtf_match_gt10
        if n_ggtf_match_gt10 else float("inf")
    )

    rows = [
        ("min-3 targets, no reconstruction cuts", stats(particle_df, None)),
        ("IDEA",      stats(particle_df, "is_reconstructable_idea")),
        ("CLD",       stats(particle_df, "is_reconstructable_cld")),
        ("displaced", stats(particle_df, "is_reconstructable_displaced")),
    ]

    md = []
    md.append(f"# CIRCE FCC-style metrics — {tag}\n")
    if tbeta is not None and td is not None:
        md.append(
            f"Operating point: `{clusterer or 'unknown'}` "
            f"`tbeta={tbeta:g}`, `td={td:g}`.\n"
        )
    else:
        md.append("Operating point: not recorded in the cache summary.\n")
    md.append(f"Total reconstructed clusters: **{n_clu:,}**.  "
              f"Mean cluster purity: **{cluster_purity:.3f}**.\n")
    md.append("")
    md.append("## Tracking efficiency (per particle, IDEA definition 1)\n")
    md.append("| Reconstructable cut | N particles | Efficiency (per hit) | Definition 1 (purity >75%) |")
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
    md.append("## Legacy selection-aware fake rate (per cluster)\n")
    md.append(
        "A candidate is fake when its majority particle is outside the named "
        "reconstructable selection or its purity does not exceed 75%. This is not the "
        "published GGTF 8% convention.\n"
    )
    md.append("| Reco set | Fake rate | 68% CI |")
    md.append("|---|---:|---|")
    md.append(f"| IDEA  | **{fr_idea:.3f}** | "
              f"+{fr_idea_hi:.3f}/-{fr_idea_lo:.3f} |")
    md.append(f"| CLD   | **{fr_cld:.3f}** | "
              f"+{fr_cld_hi:.3f}/-{fr_cld_lo:.3f} |")
    md.append("")
    md.append("## GGTF-style one-to-one assignment, candidate size >10\n")
    md.append(
        "This local reconstruction of the GGTF notebook convention counts "
        "unassigned candidates, including clones, after one-to-one assignment. "
        "The publication does not disclose enough implementation detail to "
        "guarantee exact identity.\n"
    )
    md.append("| Metric | FCC reference (no track fit) | CIRCE (this run) |")
    md.append("|---|---:|---:|")
    md.append(
        f"| GGTF ratio, >10-hit candidates (unassigned / matched) | 8.0% | "
        f"**{ggtf_ratio * 100:.1f}%** "
        f"({n_ggtf_fake_gt10:,} / {n_ggtf_match_gt10:,}) |"
    )
    md.append(
        f"| Candidate fake fraction (unassigned / all candidates) | — | "
        f"**{candidate_fr * 100:.1f}%** "
        f"(+{candidate_hi * 100:.1f}/-{candidate_lo * 100:.1f}, 68% CI; "
        f"N={n_ggtf_gt10:,}) |"
    )
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
    # Do not trust the historical convenience flag (which used >= 0.75).
    # IDEA/GGTF definition 1 is the explicit strict purity threshold.
    particle_df = particle_df.with_columns(
        (pl.col("purity_of_match") > 0.75).alias("matched")
    )
    cluster_df = pl.read_parquet(cluster_path)
    summary_path = os.path.join(args.cache_dir, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as handle:
            summary = json.load(handle)
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
        xlabel=r"Polar angle $\theta$ [degrees]",
        out_path=os.path.join(args.output_dir, "eff_vs_theta.png"),
        title="Match rate vs $\\theta$ (IDEA)",
    )
    plot_eff_vs_var(
        particle_df, "is_reconstructable_idea", "eta", eta_edges,
        xlabel=r"$\eta$",
        out_path=os.path.join(args.output_dir, "eff_vs_eta.png"),
        title="Match rate vs $\\eta$ (IDEA)",
    )
    plot_eff_vs_var(
        particle_df, "is_reconstructable_displaced", "vertex_r", vertexR_edges,
        xlabel=r"Vertex $R = \sqrt{v_x^2 + v_y^2}$ [mm]",
        out_path=os.path.join(args.output_dir, "eff_vs_vertexR.png"),
        title="Match rate vs vertex $R$ (displaced)",
        ymin=0.0,
    )
    plot_eff_vs_var(
        particle_df.with_columns(pl.col("n_hits_total").cast(pl.Float64)),
        "is_reconstructable_idea", "n_hits_total", nhit_edges,
        xlabel="Detector hits per particle",
        out_path=os.path.join(args.output_dir, "eff_vs_nhits.png"),
        title="Match rate vs $N_\\mathrm{hits}$ (IDEA)",
    )
    plot_nhit_distribution(
        particle_df,
        os.path.join(args.output_dir, "nhit_distribution.png"),
        title="Hits per particle",
    )
    plot_fake_rate_summary(
        cluster_df, particle_df,
        os.path.join(args.output_dir, "fake_rate_summary.png"),
        title="Legacy candidate fake fraction vs cluster size",
    )
    write_metrics_table(
        particle_df, cluster_df,
        os.path.join(args.output_dir, "metrics_table.md"),
        tag=args.tag,
        tbeta=summary.get("tbeta"),
        td=summary.get("td"),
        clusterer=summary.get("clusterer"),
    )

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
