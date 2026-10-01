"""Overlay match-rate vs pT for keep-all and standard configs (same ep19 model).

Reads the two fcc_unmerged/cache.parquet files, bins IDEA-reconstructable
particles in log-pT, and plots both match-rate curves with binomial 68%
(Wilson score) errors on one axis. Shared paper style. Pure CPU/IO.
"""
from __future__ import annotations
import numpy as np, polars as pl
import matplotlib.pyplot as plt
from src.eval.plotstyle import COL, yticks

# (label, config dir, colour, marker); each contributes a reconstructed curve
# (solid, from the unmerged cache) and an oracle-merge ceiling (dashed).
SERIES = [
    ("no-keep-all", "consol_nokeepall", COL["standard"], "s"),
    ("keep-all",    "consol_keepall",   COL["keepall"],  "o"),
]
EDGES = np.logspace(np.log10(0.1), np.log10(30), 26)
OUT = "/home/marko.cechovic/cgatr-paper/figures/eff_vs_pt_both"


def binned(df):
    d = df.filter(pl.col("is_reconstructable_idea"))
    pt = d["pt"].to_numpy(); m = d["matched"].to_numpy().astype(float)
    cen, rate, lo, hi = [], [], [], []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        sel = (pt >= a) & (pt < b); n = sel.sum()
        if n < 20: continue
        k = m[sel].sum(); p = k / n
        # Wilson 68% interval
        z = 1.0; z2 = z * z
        denom = 1 + z2 / n
        c = (p + z2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
        cen.append(np.sqrt(a * b)); rate.append(p); lo.append(max(0, p - (c - half))); hi.append((c + half) - p)
    return np.array(cen), np.array(rate), np.array(lo), np.array(hi)


def main():
    fig, ax = plt.subplots(figsize=(5.8, 4.1))
    for label, cfg, color, mk in SERIES:
        # reconstructed match rate (solid, markers, Wilson 68% bars)
        c, r, lo, hi = binned(pl.read_parquet(f"eval_results/{cfg}/fcc_unmerged/cache.parquet"))
        ax.errorbar(c, r, yerr=[lo, hi], fmt=f"{mk}-", color=color,
                    ms=4.5, lw=1.6, label=f"CIRCE, {label}", zorder=3)
        # oracle-merge ceiling (dashed, no markers): clusters of the same true
        # particle fused, so the gap to the solid curve is the fragmentation headroom
        co, ro, _, _ = binned(pl.read_parquet(f"eval_results/{cfg}/fcc_oracle_T0.75/cache.parquet"))
        ax.plot(co, ro, "--", color=color, lw=1.4, alpha=0.8,
                label=f"{label}, oracle merge", zorder=2)
    ax.axhline(1.0, color=COL["ref"], lw=0.8, ls=":", zorder=0)
    ax.set_xscale("log"); ax.set_xlabel(r"$p_\mathrm{T}$ [GeV]")
    ax.set_ylabel("Track match rate ($>75\\%$ purity)")
    # zoom into the high-efficiency regime where the structure lives
    ax.set_ylim(0.68, 1.015); yticks(ax, 0.05)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_title("IDEA reconstructable particles")
    # process / selection annotation, in the style of the FCC tracking talks
    ax.text(0.035, 0.06,
            "$Z\\to q\\bar q$ (uds), $\\sqrt{s}=91.2$ GeV\n"
            "$15^\\circ<\\theta<165^\\circ$,  $N_\\mathrm{hits}>10$",
            transform=ax.transAxes, fontsize=8.5, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#C9D0DE", alpha=0.9))
    for e in ("pdf", "png"):
        fig.savefig(f"{OUT}.{e}", dpi=300, bbox_inches="tight")
    print("wrote", f"{OUT}.png")


if __name__ == "__main__":
    main()
