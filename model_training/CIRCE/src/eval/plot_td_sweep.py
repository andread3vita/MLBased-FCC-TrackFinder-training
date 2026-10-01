"""Operating-point sweep figure for the paper (fig:opsweep).

Two panels from the t_d sweep (op_grid_metrics.csv + per_pt.json):
  (a) match / strict-match / hit-efficiency / fake rate vs the cluster radius t_d,
      with the singleton-inflation region shaded and the adopted t_d marked;
  (b) match-rate vs pT for every t_d in the sweep (grey family) with the adopted
      operating point highlighted — no "default" curve.

Shared paper style. Pure CPU/IO.
  python -m src.eval.plot_td_sweep --sweep_dir eval_results/ep19_op_sweep_td \
      --out /home/marko.cechovic/cgatr-paper/figures/op_sweep_td
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, polars as pl
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.eval.plotstyle import COL, SEQ_CMAP, yticks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--out", default="/home/marko.cechovic/cgatr-paper/figures/op_sweep_td")
    ap.add_argument("--adopt_tb", type=float, default=0.1)
    ap.add_argument("--adopt_td", type=float, default=0.2)
    ap.add_argument("--singleton_below", type=float, default=0.115,
                    help="shade t_d below this as the singleton-inflation region")
    a = ap.parse_args()

    df = pl.read_csv(os.path.join(a.sweep_dir, "op_grid_metrics.csv")).sort("td")
    td = df["td"].to_numpy()
    match = df["match"].to_numpy(); strict = df["match_strict"].to_numpy()
    eff = df["eff"].to_numpy(); fake = df["fake"].to_numpy()
    strict_peak = td[int(np.argmax(strict))]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.9))

    # ---- (a) metrics vs t_d ----
    axL.axvspan(td.min() - 0.005, a.singleton_below, color=COL["grey"], alpha=0.35, lw=0)
    axL.text(0.068, 0.42, "singleton\ninflation", fontsize=8, ha="center",
             color="0.35", style="italic")
    axL.plot(td, match, "o-", color=COL["match"], label="match rate")
    axL.plot(td, strict, "s-", color=COL["strict"], label="strict match")
    axL.plot(td, eff, "^-", color=COL["eff"], label="hit efficiency")
    axL.plot(td, fake, "v-", color=COL["fake"], label="fake rate")
    axL.axvline(a.adopt_td, color="0.25", ls="--", lw=1.0)
    axL.text(a.adopt_td + 0.006, 0.03, f"adopted\n$t_d={a.adopt_td:g}$", fontsize=8, color="0.25")
    axL.scatter([strict_peak], [strict.max()], marker="*", s=170,
                facecolor="none", edgecolor=COL["strict"], linewidths=1.6, zorder=5)
    axL.set_xlabel(r"Cluster radius $t_d$")
    axL.set_ylabel("Rate")
    axL.set_ylim(0, 1.0); yticks(axL, 0.1)
    axL.set_title("(a) metrics vs $t_d$ (no-keep-all set)")
    axL.legend(loc="center right", fontsize=8)

    # ---- (b) match-rate vs pT: family coloured by t_d (gradient) + adopted ----
    per_pt = json.load(open(os.path.join(a.sweep_dir, "per_pt.json")))
    adopt_key = f"tb{a.adopt_tb}_td{a.adopt_td}"
    fam = {float(k.split("_td")[1]): v for k, v in per_pt.items()}
    tds = sorted(fam)
    norm = mpl.colors.Normalize(vmin=min(tds), vmax=max(tds))
    cmap = mpl.cm.get_cmap(SEQ_CMAP)
    for d in tds:
        pts = fam[d]
        cen = [np.sqrt(lo * hi) for lo, hi, _ in pts]; r = [v for _, _, v in pts]
        if abs(d - a.adopt_td) < 1e-9:
            continue
        axR.plot(cen, r, "-", color=cmap(norm(d)), lw=1.5, alpha=0.9, zorder=2)
    if adopt_key in per_pt:
        pts = per_pt[adopt_key]
        cen = [np.sqrt(lo * hi) for lo, hi, _ in pts]; r = [v for _, _, v in pts]
        axR.plot(cen, r, "o-", color=COL["best"], lw=2.6, ms=5.5, zorder=5,
                 markeredgecolor="white",
                 label=f"adopted ($t_\\beta={a.adopt_tb:g}$, $t_d={a.adopt_td:g}$)")
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=axR, pad=0.015, fraction=0.05)
    cb.set_label(r"cluster radius $t_d$", fontsize=9)
    cb.outline.set_visible(False)
    axR.set_xscale("log")
    axR.set_xlabel(r"$p_\mathrm{T}$ [GeV]")
    axR.set_ylabel("Track match rate ($>75\\%$ purity)")
    # zoom to the populated range: every curve in the sweep sits above ~0.65
    axR.set_ylim(0.6, 1.02); yticks(axR, 0.1)
    axR.set_title("(b) match rate vs $p_\\mathrm{T}$")
    axR.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(f"{a.out}.{e}")
    print("wrote", f"{a.out}.png", "| strict peak at t_d =", strict_peak)


if __name__ == "__main__":
    main()
