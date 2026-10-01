"""Efficiency-vs-fake frontier for the fragment-merging strength.

The merge sweep never turned over: def2 efficiency rises monotonically all the
way to a merge radius of 0.50 (5x the clustering radius), so there is no
interior optimum to read off. What the sweep actually traces is a trade-off --
each step of merge strength buys def2 efficiency and pays in ghost rate -- so
the operating point has to be picked off this frontier by its slope, not by
maximising any single number. The second panel shows the knob does NOT fix the
clone population, which is the separate finding.

  python src/eval/plot_merge_frontier.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from src.eval.plotstyle import COL, plt

BASE = os.path.join(os.path.dirname(__file__), "..", "..", "eval_results")
OUT = os.path.join(BASE, "share", "merge_frontier.png")

# the two truth-free knobs, both swept at the readout OP (t_d=0.10, >=4 hits).
# merge is swept with the attach off; the attach is swept on top of merge=0.20,
# the knee of the merge frontier and the adopted operating point.
MERGE = [0.0, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
ATTACH = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
SERIES = {
    "consol_r3_nokeepall": ("Zqq (uds), 10k events", COL["standard"], "-", "o"),
    "r3_loopers":          ("Loopers, 1k events",    COL["keepall"],  "--", "s"),
}
# the draft's operating point is t_d=0.2 with no merging and no hit cut, which
# lives outside merge_sweep/ (it is the top-level eval run)
PAPER_OP = ("../fcc_unmerged", "paper OP ($t_d$=0.2, no merge)")
ADOPTED = "td0.10_mg0.20_at0_mh4"


def load(cfg, sub):
    p = os.path.join(BASE, cfg, "merge_sweep", sub, "summary.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        s = json.load(f)
    if "ghost_rate" not in s:
        return None
    return s


def sub_for(mg):
    # dirs are written with two decimals (mg0.20, not mg0.2), except plain mg0
    return f"td0.10_mg{mg:.2f}_at0_mh4" if mg else "td0.10_mg0_at0_mh4"


def sub_for_attach(at):
    return f"td0.10_mg0.20_at{at:.2f}_mh4" if at else ADOPTED


def curve(cfg, subs):
    """(knob values, ghost %, def2 %, clone %, candidates/event) for existing runs."""
    pts = [(v, load(cfg, s)) for v, s in subs]
    pts = [(v, s) for v, s in pts if s is not None]
    return (
        [v for v, _ in pts],
        [100 * s["ghost_rate"] for _, s in pts],
        [100 * s["idea"]["def2"] for _, s in pts],
        [100 * s["clone_rate"] for _, s in pts],
        [s["candidates_per_event"] for _, s in pts],
    )


fig, (axf, axc) = plt.subplots(1, 2, figsize=(13.0, 5.4))
axn = axc.twinx()
axn.grid(False)

for cfg, (label, col, ls, mk) in SERIES.items():
    mgs, g, d2, cl, cand = curve(cfg, [(v, sub_for(v)) for v in MERGE])
    if not mgs:
        continue
    axf.plot(g, d2, ls, color=col, marker=mk, label=f"{label}: merge knob", zorder=3)
    for x, y, mg in zip(g, d2, mgs):
        axf.annotate(f"{mg:g}", (x, y), textcoords="offset points", xytext=(5, -10),
                     fontsize=7.5, color=col)

    ats, ga, d2a, cla, canda = curve(cfg, [(v, sub_for_attach(v)) for v in ATTACH])
    if ats:
        axf.plot(ga, d2a, "-.", color=col, marker="^", markersize=5,
                 markerfacecolor="white",
                 label=f"{label}: attach knob (at merge 0.20)", zorder=3)

    if cfg == "consol_r3_nokeepall":
        axc.plot(mgs, cl, "-", color=col, marker=mk, label="merge knob: clone rate",
                 zorder=3)
        axn.plot(mgs, cand, ":", color=col, marker=mk, markersize=3.5, alpha=0.55,
                 label="merge knob: candidates/event", zorder=2)
        if ats:
            axc.plot(ats, cla, "-.", color=COL["fake"], marker="^",
                     label="attach knob: clone rate", zorder=3)
            axn.plot(ats, canda, ":", color=COL["fake"], marker="^", markersize=3.5,
                     alpha=0.55, label="attach knob: candidates/event", zorder=2)

    s = load(cfg, PAPER_OP[0])
    if s is not None:
        axf.plot(100 * s["ghost_rate"], 100 * s["idea"]["def2"], "*", color=col,
                 markeredgecolor=COL["ink"], markersize=15, zorder=5,
                 label=PAPER_OP[1] if cfg == "consol_r3_nokeepall" else None)
        axf.annotate(f"{s['candidates_per_event']:.0f} cand/ev,"
                     f" {s['ghost_rate'] * s['candidates_per_event']:.0f} ghosts/ev",
                     (100 * s["ghost_rate"], 100 * s["idea"]["def2"]),
                     textcoords="offset points", fontsize=7.5, color=col,
                     xytext=(10, 7) if cfg == "r3_loopers" else (10, -13))

# the knee: past merge 0.20 the exchange rate collapses from ~8 def2 points per
# ghost point to under 2
s_knee = load("consol_r3_nokeepall", ADOPTED)
if s_knee is not None:
    axf.plot(100 * s_knee["ghost_rate"], 100 * s_knee["idea"]["def2"], "o",
             color=COL["standard"], markersize=13, markerfacecolor="none",
             markeredgewidth=1.8, zorder=6, label="adopted OP (merge 0.20)")
    axf.annotate("knee: beyond merge 0.20 each ghost\npoint buys <2 def2 points, "
                 "not ~8",
                 xy=(100 * s_knee["ghost_rate"], 100 * s_knee["idea"]["def2"]),
                 xytext=(0.34, 0.95), textcoords="axes fraction", fontsize=8.5,
                 color=COL["standard"], ha="left", va="top",
                 arrowprops=dict(arrowstyle="->", color=COL["standard"], lw=1.0,
                                 connectionstyle="arc3,rad=-0.25"))

axf.axvline(8.0, color=COL["ref"], lw=1.0, ls=":", zorder=1)
axf.annotate("GGTF quotes 8% fake\n(formula not published)", xy=(8.0, 74.6),
             xytext=(0.03, 0.13), textcoords="axes fraction", fontsize=8,
             color=COL["ref"], ha="left", va="bottom",
             arrowprops=dict(arrowstyle="-", color=COL["ref"], lw=0.8, ls=":"))
axf.set_xlabel("Ghost rate [%]   (no particle above 75% of the candidate's hits)")
axf.set_ylabel("Efficiency, definition 2 [%]")
axf.set_title("Merging buys def2 efficiency and pays in ghosts", loc="left")
axf.legend(loc="lower right", fontsize=8)

axc.set_xlabel("Knob radius   (merge radius, or attach radius at merge 0.20)")
axc.set_ylabel("Clone rate [%]")
axn.set_ylabel("Candidates / event", color=COL["ref"])
axn.tick_params(axis="y", labelcolor=COL["ref"])
axc.set_title("Only the attach reduces clones and candidates together (Zqq)",
              loc="left")
axc.set_ylim(0, 100)
axn.set_ylim(0, 160)
h1, l1 = axc.get_legend_handles_labels()
h2, l2 = axn.get_legend_handles_labels()
axc.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.5)

fig.text(0.5, -0.04,
         "All points at $t_\\beta$=0.1, $t_d$=0.10 with the $\\geq$4-hit track-candidate "
         "requirement; labels on the left panel are the merge radius, and the attach "
         "series is swept on top of merge=0.20. def2 is GGTF "
         "efficiency definition 2 (purity > 50% AND hit efficiency > 50%) on "
         "IDEA-reconstructable particles, reported here instead of our match rate\n"
         "because the match rate credits a pure fragment of a track as reconstructed. "
         "def2 never turns over, so there is no interior optimum: the operating point "
         "has to be read off the slope. The draft's operating point looks better on "
         "ghost RATE only because it emits 609 candidates per event: in absolute terms "
         "it puts 34.7 ghost tracks in each event against 12.0 at the adopted point.",
         ha="center", va="top", fontsize=8, color=COL["ref"])

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.abspath(OUT))
