"""Production-vertex-radius diagnostic (fig:vertexr).

Answers "is the efficiency drop at large vertex R just low statistics, or real?"
by showing, per R bin: the number of IDEA-reconstructable particles (bars, log
scale) and the match rate (line). Reads a per-particle cache.parquet with
`vertex_r`, `is_reconstructable_idea`, `matched`. Shared paper style. CPU/IO.
  python -m src.eval.plot_vertexr_dist --cache eval_results/ep19_keepall/fcc_unmerged/cache.parquet
"""
from __future__ import annotations
import argparse
import numpy as np, polars as pl
import matplotlib.pyplot as plt
from src.eval.plotstyle import COL, yticks

EDGES = [0, 5, 20, 50, 100, 200, 400, 800, 1500, 3000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="/home/marko.cechovic/cgatr-paper/figures/vertexr_distribution")
    a = ap.parse_args()

    d = pl.read_parquet(a.cache).filter(pl.col("is_reconstructable_idea"))
    r = d["vertex_r"].to_numpy(); m = d["matched"].to_numpy().astype(float)
    ntot = len(d)

    cnt, rate, lo, hi, labels = [], [], [], [], []
    from scipy.stats import binomtest
    for a0, b0 in zip(EDGES[:-1], EDGES[1:]):
        s = (r >= a0) & (r < b0); n = int(s.sum())
        if n == 0:
            continue
        p = m[s].mean()
        ci = binomtest(int(m[s].sum()), n).proportion_ci(0.68)
        cnt.append(n); rate.append(p); lo.append(p - ci.low); hi.append(ci.high - p)
        labels.append(f"{a0}–{b0}")
    x = np.arange(len(cnt))

    fig, axL = plt.subplots(figsize=(6.2, 4.0))
    # distribution: particle counts per bin (log scale, bars)
    axL.bar(x, cnt, width=0.72, color=COL["grey"], edgecolor="white", linewidth=0.8,
            zorder=2, label="particles per bin")
    axL.set_yscale("log")
    axL.set_ylim(min(cnt) * 0.6, max(cnt) * 6)  # headroom so the tallest bar clears the title
    axL.set_ylabel("IDEA-reconstructable particles / bin")
    axL.set_xlabel("Production-vertex radius $R=\\sqrt{v_x^2+v_y^2}$ [mm]")
    axL.set_xticks(x); axL.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    axL.grid(axis="y", alpha=0.4)
    for xi, n in zip(x, cnt):
        axL.text(xi, n * 1.25, f"{100*n/ntot:.0f}%", ha="center", va="bottom",
                 fontsize=7.5, color="#5C6883")

    # match rate on a twin axis
    axR = axL.twinx(); axR.grid(False)
    axR.errorbar(x, rate, yerr=[lo, hi], fmt="o-", color=COL["keepall"],
                 ms=5, lw=1.8, zorder=4, label="match rate")
    axR.set_ylabel("Track match rate ($>75\\%$ purity)", color=COL["keepall"])
    axR.tick_params(axis="y", labelcolor=COL["keepall"])
    axR.set_ylim(0.5, 1.0); yticks(axR, 0.1)

    # combined legend, placed upper-right where the match rate has dropped away
    h1, l1 = axL.get_legend_handles_labels(); h2, l2 = axR.get_legend_handles_labels()
    axR.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.5)
    axL.set_title("Vertex-radius distribution and match rate (IDEA, keep-all)", pad=10)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(f"{a.out}.{e}")
    print("wrote", f"{a.out}.png")


if __name__ == "__main__":
    main()
