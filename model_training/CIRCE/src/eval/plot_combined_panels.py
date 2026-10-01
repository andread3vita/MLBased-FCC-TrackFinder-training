"""Combine the two-panel paper figures into a SINGLE input image file each, as
required by the Springer Nature user manual (section 7.3: "make sure that each
figure is from a single input image file. Avoid using subfigures.").

Produces:
  eff_vs_theta_nhits.{pdf,png}  -- match rate vs theta (a) and vs n_hits (b),
                                    a genuine single vector figure from the cache.
  embed_support.{pdf,png}       -- PCA scree (a) + t-SNE grid (b) composited from
                                    the already-rendered panels (no t-SNE re-run).
Pure CPU/IO.  Usage:
  python -m src.eval.plot_combined_panels --cache eval_results/ep19_keepall/fcc_unmerged/cache.parquet
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, polars as pl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src.eval.plot_fcc_metrics import binned_efficiency
from src.eval.plotstyle import COL, yticks

FIG = "/home/marko.cechovic/cgatr-paper/figures"
PCA_JSON = "/home/marko.cechovic/cgatr/model_training/eval_results/v35_analysis_merged/pca_results.json"


def scree(pca_json=PCA_JSON):
    """Redraw the 5-D design-model PCA scree natively (shared style, capitalised
    axis labels), so its png/pdf match and are consistent with the other plots."""
    j = json.load(open(pca_json))
    evr = np.array(j["explained_variance_ratio"]) * 100
    d = len(evr)
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.bar(range(1, d + 1), evr, color=COL["standard"], alpha=0.85)
    ax.plot(range(1, d + 1), np.cumsum(evr), "o-", color=COL["cum"], label="cumulative")
    for i, v in enumerate(evr):
        ax.text(i + 1, v + 1, f"{v:.1f}%" if v >= 0.05 else "~0%", ha="center", fontsize=8)
    ax.set_xlabel("Principal component"); ax.set_ylabel("Variance explained (%)")
    ax.set_xticks(range(1, d + 1)); ax.set_ylim(0, 105); ax.legend()
    ax.set_title(f"PCA of the {d}-D design embedding")
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(f"{FIG}/embed_pca_scree.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote embed_pca_scree.{pdf,png}")


def _panel(ax, centers, p, lo, hi, xlabel, title, logx=False, ymin=0.5):
    ax.errorbar(centers, p, yerr=[lo, hi], fmt="o-", color=COL["keepall"],
                ecolor=COL["keepall"], markersize=4.5, linewidth=1.6, label="CIRCE")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Track match rate ($>75\\%$ purity)")
    ax.set_ylim(ymin, 1.05); yticks(ax, 0.1)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title(title)
    ax.legend(loc="lower right")


def eff_theta_nhits(cache):
    d = pl.read_parquet(cache).filter(pl.col("is_reconstructable_idea"))
    theta_edges = list(np.linspace(15, 165, 11))
    nhit_edges = [4, 7, 10, 15, 20, 30, 50, 80, 130, 200, 350, 600]
    ct, pt_, lt, ht, _ = binned_efficiency(d, "theta_deg", theta_edges, "matched")
    cn, pn, ln, hn, _ = binned_efficiency(d, "n_hits_total", nhit_edges, "matched")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.9))
    _panel(axL, ct, pt_, lt, ht, r"Polar angle $\theta$ [degrees]",
           r"(a) Match rate vs $\theta$")
    _panel(axR, cn, pn, ln, hn, "Detector hits per particle",
           r"(b) Match rate vs $N_\mathrm{hits}$")
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(f"{FIG}/eff_vs_theta_nhits.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote eff_vs_theta_nhits.{pdf,png}")


def embed_support():
    """Composite the existing scree + t-SNE renders into one image file, so the
    approved look is preserved without re-running the (expensive) t-SNE."""
    scree = mpimg.imread(f"{FIG}/embed_pca_scree.png")
    tsne = mpimg.imread(f"{FIG}/embed_final_tsne.png")
    # width ratio from the native pixel widths so neither panel is distorted
    wr = [scree.shape[1], tsne.shape[1]]
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11.0, 11.0 * max(scree.shape[0] / scree.shape[1],
                                        tsne.shape[0] / tsne.shape[1]) * 0.62),
        gridspec_kw={"width_ratios": wr})
    for ax, im, lab in ((axL, scree, "(a)"), (axR, tsne, "(b)")):
        ax.imshow(im)
        ax.axis("off")
        ax.text(0.02, 0.98, lab, transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="top", ha="left")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
    for e in ("pdf", "png"):
        fig.savefig(f"{FIG}/embed_support.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote embed_support.{pdf,png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="eval_results/ep19_keepall/fcc_unmerged/cache.parquet")
    a = ap.parse_args()
    eff_theta_nhits(a.cache)
    scree()
    embed_support()


if __name__ == "__main__":
    main()
