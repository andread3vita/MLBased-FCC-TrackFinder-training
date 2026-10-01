"""What the ghost rate counts, and what our fake rate counts on top of it.

Every emitted track candidate falls into exactly one of four categories, so the
two fake conventions are just different subsets of the same bar:

  unique good       first >=75%-pure match to an IDEA-reconstructable particle
  clone             a further >=75%-pure match to a particle already matched
  clean non-selected  >=75% pure, but the particle fails the IDEA selection
                    (out of 15-165 deg, <=10 hits, secondary, or neutral)
  ghost             no particle contributes >=75% of the candidate's hits

  ghost rate      = ghost / all candidates            <- the standard convention
  our fake rate   = (ghost + clean non-selected) / all candidates

  python src/eval/plot_fake_definitions.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import polars as pl

from src.eval.plotstyle import COL, plt

BASE = os.path.join(os.path.dirname(__file__), "..", "..", "eval_results")
OUT = os.path.join(BASE, "share", "fake_definitions.png")

CASES = [
    ("Zqq, draft OP\n$t_d$=0.2, no merging", "consol_r3_nokeepall", "fcc_unmerged", 10000),
    ("Zqq, adopted\nmerge 0.20, $\\geq$4 hits", "consol_r3_nokeepall",
     "merge_sweep/td0.10_mg0.20_at0_mh4", 10000),
    ("Loopers, draft OP\n$t_d$=0.2, no merging", "r3_loopers", "fcc_unmerged", 1000),
    ("Loopers, adopted\nmerge 0.20, $\\geq$4 hits", "r3_loopers",
     "merge_sweep/td0.10_mg0.20_at0_mh4", 1000),
]
CATS = [("unique good track", COL["eff"]), ("clone", COL["grey"]),
        ("clean, particle not selected", COL["fake"]), ("ghost", COL["strict"])]


def decompose(cfg, sub, n_events):
    c = pl.read_parquet(os.path.join(BASE, cfg, sub, "cache_clusters.parquet"),
                        columns=["matched_mc_idx", "event_id", "seed", "purity",
                                 "is_fake_idea"])
    ghosts = int((c["purity"] < 0.75).sum())
    good = c.filter(~pl.col("is_fake_idea"))
    uniq = len(good.group_by(["matched_mc_idx", "event_id", "seed"]).len())
    return {
        "n": len(c),
        "cand": len(c) / n_events,
        "parts": np.array([uniq, len(good) - uniq, len(c) - ghosts - len(good), ghosts]),
    }


rows = [(lab, decompose(cfg, sub, n)) for lab, cfg, sub, n in CASES]

fig, (axc, axf) = plt.subplots(1, 2, figsize=(13.4, 5.0),
                               gridspec_kw={"width_ratios": [1.55, 1.0]})
y = np.arange(len(rows))[::-1]

for i, (lab, d) in enumerate(rows):
    frac = 100 * d["parts"] / d["n"]
    left = 0.0
    for (cname, col), f in zip(CATS, frac):
        axc.barh(y[i], f, left=left, color=col, edgecolor="white", linewidth=0.8,
                 label=cname if i == 0 else None, height=0.62)
        if f > 6:
            axc.text(left + f / 2, y[i], f"{f:.0f}%", ha="center", va="center",
                     fontsize=9, color="white", fontweight="bold")
        left += f
    axc.text(101.5, y[i], f"{d['cand']:.0f} cand/ev", va="center", fontsize=8.5,
             color=COL["ref"])

axc.set_yticks(y)
axc.set_yticklabels([lab for lab, _ in rows], fontsize=9)
axc.set_xlabel("Share of emitted track candidates [%]")
axc.set_xlim(0, 100)
axc.set_title("Every candidate is one of four things", loc="left")
axc.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=8.5)
axc.grid(axis="y", visible=False)

# the same two right-hand categories, replotted as the two conventions
w = 0.38
xs = np.arange(len(rows))
ghost = np.array([100 * d["parts"][3] / d["n"] for _, d in rows])
extra = np.array([100 * d["parts"][2] / d["n"] for _, d in rows])

axf.bar(xs - w / 2, ghost, w, color=COL["strict"], label="ghost rate (standard)")
axf.bar(xs + w / 2, ghost, w, color=COL["strict"])
axf.bar(xs + w / 2, extra, w, bottom=ghost, color=COL["fake"],
        label="+ clean, particle not selected  =  our fake rate")
for x, g, e in zip(xs, ghost, extra):
    axf.text(x - w / 2, g + 0.4, f"{g:.1f}", ha="center", fontsize=8.5,
             color=COL["strict"], fontweight="bold")
    axf.text(x + w / 2, g + e + 0.4, f"{g + e:.1f}", ha="center", fontsize=8.5,
             color=COL["ink"], fontweight="bold")

axf.axhline(8.0, color=COL["ref"], lw=1.2, ls="--")
axf.annotate("GGTF quotes 8%\n(which of these two?)", xy=(0.42, 8.0),
             xytext=(0.06, 0.63), textcoords="axes fraction", fontsize=8.5,
             color=COL["ref"], ha="left",
             arrowprops=dict(arrowstyle="->", color=COL["ref"], lw=1.0))
axf.set_xticks(xs)
axf.set_xticklabels(["Zqq\ndraft OP", "Zqq\nadopted", "Loopers\ndraft OP",
                     "Loopers\nadopted"], fontsize=8.5)
axf.set_ylabel("Rate [%]")
axf.set_ylim(0, 21)
axf.set_title("The same output, two conventions", loc="left")
axf.legend(loc="upper left", fontsize=8.5)

fig.text(0.5, -0.10,
         "A candidate is a ghost when no single true particle contributes at least 75% "
         "of its hits; this is the convention LHCb and Belle II report. Our fake rate "
         "adds candidates that are clean reconstructions of a real particle which fails "
         "the IDEA\nreconstructable selection (outside 15-165 degrees, 10 or fewer hits, "
         "secondary, or neutral). Clones - extra clean candidates for a particle already "
         "reconstructed - are counted by neither convention, and they are the majority "
         "of everything we emit.",
         ha="center", va="top", fontsize=8.5, color=COL["ref"])

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.abspath(OUT))
