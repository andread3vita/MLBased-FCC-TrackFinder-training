"""Efficiency vs pT under all four GGTF trackingEfficiencyPlot denominators.

Matching and binning are GGTF's own: a target counts as reconstructed when at
least one candidate matches it at purity > 0.75; bins are
exp(arange(log 0.1, log 60, 0.2)) with binomial errors capped at 1. The four
curves are the 2x2 of their notebook call-site options, genStatus [1] or [0,1]
x nHits > 3 or > 10, all at 15 < theta < 165 and pT > 0.

Clustering: beta-greedy, default tbeta 0.6, td 0.3, no candidate hit cut.
Defaults reproduce the epoch-24 Loopers preview; point --dir at another eval
directory (holding emb/forward_hits.parquet and mc_signal.parquet) and set
--label to plot a different checkpoint.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/home/marko.cechovic/cgatr-paper/literature/cgatr_fcc_pkg")
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eval.fcc_cache_parallel import cluster_event as apply_clusterer
from src.eval.ggtf_clustering_sweep import load_events
from src.eval.plotstyle import COL

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--dir", default="/home/marko.cechovic/cgatr-runs/"
                                 "e3-paper-physical/loopers_preview",
                help="eval dir with emb/forward_hits.parquet + mc_signal.parquet")
ap.add_argument("--label", default="epoch-24 checkpoint",
                help="checkpoint description for the figure title")
ap.add_argument("--sample-name", default="Loopers",
                help="sample name for the title; the low-pT footnote is "
                     "Loopers-specific and only printed for that sample")
ap.add_argument("--tbeta", type=float, default=0.6)
ap.add_argument("--td", type=float, default=0.3)
ap.add_argument("--rows", type=Path, default=None,
                help="precomputed target_rows.parquet from ggtf_full_eval; "
                     "skips the forward cache and clustering entirely")
ap.add_argument("--n-events", type=int, default=None,
                help="event count for the title when --rows is used")
args = ap.parse_args()

L = args.dir
OUT = Path(f"{L}/ggtf_exact")
OUT.mkdir(parents=True, exist_ok=True)
TBETA, TD = args.tbeta, args.td
MIN_N = 8

if args.rows is not None:
    df = pl.read_parquet(args.rows).filter(
        (pl.col("pt") > 0) & (pl.col("theta_deg") > 15) & (pl.col("theta_deg") < 165))
    rows = list(df.select(["nhits", "gen_status", "pt", "matched"]).iter_rows())
    n_events = args.n_events if args.n_events is not None else df["event_id"].n_unique()
else:
    events = load_events(Path(f"{L}/emb/forward_hits.parquet"), 4)
    n_events = None
mc = None if args.rows is not None else pl.read_parquet(f"{L}/mc_signal.parquet",
                     columns=["mc_index", "event_id", "seed",
                              "gen_status", "pt", "theta"])
if args.rows is None:
    truth = {(int(r["seed"]), int(r["event_id"]), int(r["mc_index"])):
             (r["gen_status"], r["pt"], np.degrees(r["theta"]))
             for r in mc.iter_rows(named=True)}
    rows = []
if args.rows is None:
  for ev in events:
    labels = np.asarray(apply_clusterer(
        "greedy", ev["betas"], ev["coords"], TBETA, TD, 0))
    arr = ev["mc"]
    uniq, counts = np.unique(arr, return_counts=True)
    nhits = {int(p): int(c) for p, c in zip(uniq, counts)}
    targets = {p for p, c in nhits.items() if c >= 3}
    csize = defaultdict(int)
    for lab in labels[labels >= 0]:
        csize[int(lab)] += 1
    overlap = defaultdict(int)
    sel = labels >= 0
    for p, lab in zip(arr[sel], labels[sel]):
        if int(p) in targets:
            overlap[(int(p), int(lab))] += 1
    matched = {p for (p, lab), n in overlap.items() if n / csize[lab] > 0.75}
    for p in targets:
        gs, pt, th = truth.get((ev["seed"], ev["event_id"], p),
                               (None, None, None))
        if gs is None or pt is None or not pt > 0 or not 15 < th < 165:
            continue
        rows.append((nhits[p], gs, pt, p in matched))
  n_events = len(events)

bins = np.exp(np.arange(np.log(0.1), np.log(60), 0.2))     # their binning

PANELS = [
    ("genStatus = [1]  (generator particles only)", {1}),
    ("genStatus = [0, 1]  (secondaries included)", {0, 1}),
]
CUTS = [
    ("nHits > 3",  3,  COL["standard"], "-",  "s"),
    ("nHits > 10", 10, COL["fake"],     "--", "o"),
]


def curve(gens, nmin):
    pts = np.array([pt for nh, gs, pt, m in rows if gs in gens and nh > nmin])
    ms = np.array([m for nh, gs, pt, m in rows if gs in gens and nh > nmin],
                  dtype=float)
    idx = np.digitize(pts, bins)
    xs, ys, es = [], [], []
    for i in range(1, len(bins)):
        sel = idx == i
        n = int(sel.sum())
        xs.append((bins[i - 1] + bins[i]) / 2)               # their centres
        if n < MIN_N:
            ys.append(np.nan)                                # break the line
            es.append(0.0)
            continue
        eff = ms[sel].mean()
        ys.append(eff)
        es.append(np.sqrt(eff * (1 - eff) / n))
    ys = np.array(ys); es = np.array(es)
    return xs, ys, es, np.minimum(ys + es, 1.0) - ys         # their cap at 1


curves = {(title, label): curve(gens, nmin)
          for title, gens in PANELS for label, nmin, _c, _l, _m in CUTS}
finite = np.concatenate([ys[np.isfinite(ys)] for _, ys, _, _ in curves.values()])
# Zoomed floor only when every visible bin sits above it; a model that loses a
# particle class (e.g. trained without secondaries) has near-zero bins that a
# fixed 0.35 floor would silently push out of frame.
ylo = 0.35 if finite.min() >= 0.4 else -0.02

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
for ax, (title, gens) in zip(axes, PANELS):
    for label, nmin, color, ls, mk in CUTS:
        xs, ys, lo, hi = curves[(title, label)]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=f"{mk}{ls}", color=color,
                    ms=4.6, lw=1.7, capsize=3, label=label)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 12)
    ax.set_ylim(ylo, 1.02)
    ax.set_xlabel(r"$p_\mathrm{T}$ [GeV]")
    ax.set_title(title, fontsize=10.5)
    ax.grid(which="major", linestyle=":", linewidth=0.6)
    ax.legend(loc="lower right", fontsize=9, title="denominator hit cut",
              title_fontsize=9)
axes[0].set_ylabel("Tracking efficiency")
fig.suptitle(f"{args.sample_name}, {n_events} events, {args.label}: efficiency under "
             "each trackingEfficiencyPlot denominator (matching purity > 0.75; "
             + r"greedy $t_\beta$=" + f"{TBETA:g}" + r", $t_d$=" + f"{TD:g}"
             + ", no candidate cut)", fontsize=11)
foot = ("Same model, same candidates in all four curves; the spread is the "
        "denominator definition alone. 15 < theta < 165, pT > 0; bins "
        "exp(0.1..60, step 0.2); binomial errors; lines break where a bin has "
        "fewer than 8 targets.")
if args.sample_name == "Loopers":
    foot += ("\nNote the left panel is empty below ~2.5 GeV: this sample's "
             "generator-status-1 particles are all at high pT, so the low-pT "
             "loopers appear only on the right.")
fig.text(0.09, -0.03, foot, fontsize=7.8, va="top")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(OUT / f"eff_vs_pt_ggtf_denominators_loopers.{e}", dpi=200,
                bbox_inches="tight")
print("wrote", OUT / "eff_vs_pt_ggtf_denominators_loopers.png")
