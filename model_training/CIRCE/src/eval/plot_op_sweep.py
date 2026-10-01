"""Plot the (tbeta, td) operating-point sweep: metric heatmaps + best-OP curves.

Reads op_grid_metrics.csv (tbeta, td, match, eff, purity, fake) and per_pt.json
from sweep_parallel, renders heatmaps of match rate / fake rate / efficiency
over the grid (best match-rate OP marked), and the match-rate-vs-pT curves for
the best OP and the paper default. Pure CPU/IO.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, polars as pl
import matplotlib.pyplot as plt
from src.eval.plotstyle import COL


def heat(ax, tb, td, vals, title, fmt="{:.2f}", cmap="viridis", star=None):
    T = sorted(set(tb)); D = sorted(set(td))
    Z = np.full((len(T), len(D)), np.nan)
    for b, d, v in zip(tb, td, vals):
        Z[T.index(b), D.index(d)] = v
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(D))); ax.set_xticklabels([f"{x:g}" for x in D])
    ax.set_yticks(range(len(T))); ax.set_yticklabels([f"{x:g}" for x in T])
    ax.set_xlabel("$t_d$"); ax.set_ylabel(r"$t_\beta$"); ax.set_title(title, fontsize=10)
    for i in range(len(T)):
        for j in range(len(D)):
            if not np.isnan(Z[i, j]):
                ax.text(j, i, fmt.format(Z[i, j]), ha="center", va="center",
                        fontsize=7, color="w")
    if star is not None:
        ax.scatter([D.index(star[1])], [T.index(star[0])], marker="*", s=260,
                   edgecolor="red", facecolor="none", linewidths=1.8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--out", default="/home/marko.cechovic/cgatr-paper/figures/op_sweep")
    ap.add_argument("--def_tb", type=float, default=0.1)
    ap.add_argument("--def_td", type=float, default=0.2)
    a = ap.parse_args()

    df = pl.read_parquet(a.sweep_dir + "/op_grid_metrics.parquet") if os.path.exists(
        a.sweep_dir + "/op_grid_metrics.parquet") else pl.read_csv(a.sweep_dir + "/op_grid_metrics.csv")
    tb = df["tbeta"].to_list(); td = df["td"].to_list()
    match = df["match"].to_list(); fake = df["fake"].to_list(); eff = df["eff"].to_list()

    best_i = int(np.argmax(match))
    best = (tb[best_i], td[best_i])
    print(f"best match-rate OP: tbeta={best[0]} td={best[1]}  "
          f"match={match[best_i]:.3f} fake={fake[best_i]:.3f} eff={eff[best_i]:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    heat(axes[0], tb, td, match, "IDEA match rate (>75% purity)", cmap="viridis", star=best)
    heat(axes[1], tb, td, fake, "IDEA fake rate", cmap="magma_r", star=best)
    heat(axes[2], tb, td, eff, "Track hit efficiency (matched)", cmap="viridis", star=best)
    fig.suptitle(r"Operating-point sweep on the epoch-19 model (\star = best match rate)".replace("\\star", "*"),
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for e in ("pdf", "png"):
        fig.savefig(f"{a.out}_heatmap.{e}", dpi=150, bbox_inches="tight")
    print("wrote", f"{a.out}_heatmap.png")

    # eff-vs-pt curves: the whole operating-point family in grey, adopted highlighted
    per_pt = json.load(open(a.sweep_dir + "/per_pt.json"))
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    adopt_key = f"tb{a.def_tb}_td{a.def_td}"
    for key, pts in per_pt.items():
        if key == adopt_key:
            continue
        cen = [np.sqrt(lo * hi) for lo, hi, _ in pts]; r = [v for _, _, v in pts]
        ax.plot(cen, r, "-", color=COL["grey"], lw=1.0, alpha=0.9, zorder=1)
    if adopt_key in per_pt:
        pts = per_pt[adopt_key]
        cen = [np.sqrt(lo * hi) for lo, hi, _ in pts]; r = [v for _, _, v in pts]
        ax.plot(cen, r, "o-", color=COL["best"], ms=4.5, lw=1.8, zorder=3,
                label=f"adopted ($t_\\beta$={a.def_tb:g}, $t_d$={a.def_td:g})")
    ax.plot([], [], "-", color=COL["grey"], lw=1.0, label="other operating points")
    ax.set_xscale("log"); ax.set_xlabel(r"$p_\mathrm{T}$ [GeV]")
    ax.set_ylabel("Track match rate ($>75\\%$ purity)"); ax.set_ylim(0, 1.03)
    ax.legend(loc="lower right"); ax.set_title("IDEA reconstructable")
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(f"{a.out}_eff_vs_pt.{e}", dpi=150, bbox_inches="tight")
    print("wrote", f"{a.out}_eff_vs_pt.png")


if __name__ == "__main__":
    main()
