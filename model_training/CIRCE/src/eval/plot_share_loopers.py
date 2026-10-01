"""Loopers eval dataset (all 1000 seeds, keep-all truth): one figure, four
panels vs pT - track match rate, track hit efficiency, fake rate and fake
tracks per event - for the five reconstruction variants at t_beta=0.1, t_d=0.10:
CIRCE, CIRCE with the >=4-hit track-candidate requirement, CIRCE + fragment
merging, CIRCE + merging + >=4 hits, and the oracle-merge ceiling.

The fake rate is a FRACTION OF EMITTED CLUSTERS, so the >=4-hit requirement
raises it (it removes 12x more clusters than fakes) while cutting the absolute
number of fake tracks per event by ~9x.  Both quantities are therefore plotted,
and the set-level numbers for every variant are tabulated on the figure.
"""
import json
import os
import sys

sys.path.insert(0, "/home/marko.cechovic/cgatr-paper/literature/cgatr_fcc_pkg")
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from src.eval.plotstyle import COL

BASE = "/home/marko.cechovic/cgatr-paper/literature/cgatr_fcc_pkg/eval_results/r3_loopers/merge_sweep"
OUT = "/home/marko.cechovic/cgatr-paper/literature/cgatr_fcc_pkg/eval_results/share"
os.makedirs(OUT, exist_ok=True)
EDGES = np.logspace(np.log10(0.1), np.log10(10), 13)
MIN_N = 40
N_EVENTS = 1000

# The Loopers sample generates particles up to 10 GeV over the full solid angle,
# so clusters from particles just outside 15deg < theta < 165deg pile up below
# pT = 10 GeV * sin(15deg) = 2.59 GeV and count as fake under the IDEA selection.
OOA_BAND = (1.0, 2.7)

SERIES = [
    ("td0.10_mg0_at0",              "CIRCE",                          COL["standard"], "-",  "s", 0.94),
    ("td0.10_mg0_at0_mh4",          "CIRCE, $\\geq$4 hits",           COL["eff"],      "-",  "v", 0.97),
    ("td0.10_mg0.15_at0",           "CIRCE + fragment merging",       COL["fake"],     "-",  "^", 1.00),
    ("td0.10_mg0.15_at0_mh4",       "CIRCE + merging, $\\geq$4 hits", COL["ink"],      "-",  "D", 1.03),
    ("td0.10_mg0_at0_oracle_T0.75", "CIRCE, oracle merge",            COL["keepall"],  "--", "o", 1.06),
]


def wilson(k, n):
    p = k / n
    z2 = 1.0
    denom = 1 + z2 / n
    c = (p + z2 / (2 * n)) / denom
    half = np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return p, max(0, p - (c - half)), max(0, (c + half) - p)


def tracks_binned(sub):
    d = (pl.read_parquet(f"{BASE}/{sub}/cache.parquet",
                         columns=["pt", "matched", "efficiency_per_hit",
                                  "is_reconstructable_idea"])
           .filter(pl.col("is_reconstructable_idea")))
    pt = d["pt"].to_numpy()
    m = d["matched"].to_numpy().astype(float)
    e = d["efficiency_per_hit"].to_numpy()
    rows = []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        sel = (pt >= a) & (pt < b); n = int(sel.sum())
        if n < MIN_N:
            continue
        p, lo, hi = wilson(m[sel].sum(), n)
        eh = e[sel]
        rows.append((np.sqrt(a * b), p, lo, hi, eh.mean(), eh.std() / np.sqrt(n)))
    return np.array(rows)


def fakes_binned(sub):
    """Per-pT fake rate and fake count, under both fake definitions.

    purity:   the candidate has no true particle contributing >= 75% of its
              hits, i.e. it does not correspond to a single trajectory.
    fiducial: the above, OR the candidate's majority particle fails the IDEA
              reconstructable selection (theta outside 15-165 deg, <= 10 hits).
              Clean reconstructions of real particles just outside the fiducial
              region are counted as fake here, which is what produces the
              1-2.7 GeV spike on this sample.

    pT is the majority particle's, so only clusters with a particle majority
    enter (noise-majority clusters have no pT and reach the set-level numbers
    only)."""
    d = pl.read_parquet(f"{BASE}/{sub}/cache_clusters.parquet",
                        columns=["pt", "purity", "is_fake_idea"]).drop_nulls("pt")
    pt = d["pt"].to_numpy()
    imp = (d["purity"].to_numpy() < 0.75).astype(float)
    fid = d["is_fake_idea"].to_numpy().astype(float)
    rows = []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        sel = (pt >= a) & (pt < b); n = int(sel.sum())
        if n < MIN_N:
            continue
        k = imp[sel].sum()
        p, lo, hi = wilson(k, n)
        rows.append((np.sqrt(a * b), p, lo, hi, k / N_EVENTS, np.sqrt(k) / N_EVENTS,
                     fid[sel].mean()))
    return np.array(rows)


def totals(sub):
    """Set-level numbers: match rate, hit efficiency, candidates/event, clones
    per event, both fake rates and fake tracks/event. The oracle merge is a
    truth-based diagnostic, so its cluster population - and hence any fake or
    clone number - is not meaningful."""
    d = (pl.read_parquet(f"{BASE}/{sub}/cache.parquet",
                         columns=["matched", "efficiency_per_hit",
                                  "is_reconstructable_idea"])
           .filter(pl.col("is_reconstructable_idea")))
    out = {"match": float(d["matched"].cast(pl.Float64).mean()),
           "hiteff": float(d["efficiency_per_hit"].mean())}
    c = pl.read_parquet(f"{BASE}/{sub}/cache_clusters.parquet",
                        columns=["matched_mc_idx", "event_id", "seed",
                                 "purity", "is_fake_idea"])
    out["cand"] = len(c) / N_EVENTS
    if "oracle" in sub:
        out.update(fake_p=None, fake_f=None, fakes=None, clones=None)
        return out
    n_imp = int((c["purity"] < 0.75).sum())
    out["fake_p"] = n_imp / len(c)
    out["fake_f"] = json.load(open(f"{BASE}/{sub}/summary.json"))["fake_rate_idea"]
    out["fakes"] = n_imp / N_EVENTS
    # clones: extra >=75%-pure candidates for a particle that another candidate
    # already reconstructs. They are not fakes under either definition.
    good = c.filter((pl.col("purity") >= 0.75) & ~pl.col("is_fake_idea"))
    n_uniq = len(good.group_by(["matched_mc_idx", "event_id", "seed"]).len())
    out["clones"] = (len(good) - n_uniq) / N_EVENTS
    return out


fig, (axm, axe, axf, axn) = plt.subplots(
    4, 1, figsize=(7.4, 12.0), sharex=True,
    gridspec_kw={"hspace": 0.09, "height_ratios": [1.0, 1.0, 1.15, 1.0]})

rows = []
for sub, label, color, ls, mk, xoff in SERIES:
    t = tracks_binned(sub)
    axm.errorbar(t[:, 0] * xoff, 100 * t[:, 1], yerr=[100 * t[:, 2], 100 * t[:, 3]],
                 fmt=f"{mk}{ls}", color=color, ms=4.2, lw=1.7, label=label)
    axe.errorbar(t[:, 0] * xoff, 100 * t[:, 4], yerr=100 * t[:, 5],
                 fmt=f"{mk}{ls}", color=color, ms=4.2, lw=1.7)
    if "oracle" not in sub:
        f = fakes_binned(sub)
        axf.errorbar(f[:, 0] * xoff, 100 * f[:, 1], yerr=[100 * f[:, 2], 100 * f[:, 3]],
                     fmt=f"{mk}{ls}", color=color, ms=4.2, lw=1.7)
        axn.errorbar(f[:, 0] * xoff, f[:, 4], yerr=f[:, 5],
                     fmt=f"{mk}{ls}", color=color, ms=4.2, lw=1.7)
        if sub == "td0.10_mg0.15_at0_mh4":       # what the fiducial clause adds
            axf.plot(f[:, 0], 100 * f[:, 6], ":", color=COL["ref"], lw=1.4,
                     marker="x", ms=4, mew=1.0,
                     label="same recipe, counting clean tracks of\n"
                           "out-of-fiducial particles as fake")
    rows.append((label, totals(sub)))

axm.set_ylabel("Track match rate [%]\n($>75\\%$ purity)")
axm.set_ylim(74, 101)
axm.yaxis.set_major_locator(MultipleLocator(5))
axm.legend(loc="lower right", fontsize=9, framealpha=0.95)
axm.set_title("IDEA_v4_o1 Loopers sample, 1000 events, keep-all truth  "
              "($t_\\beta=0.1$, $t_d=0.10$)")

axe.set_ylabel("Track hit efficiency [%]")
axe.set_ylim(0, 105)
axe.yaxis.set_major_locator(MultipleLocator(20))

axf.axvspan(*OOA_BAND, color=COL["grey"], alpha=0.28, lw=0, zorder=0)
axf.annotate("the whole spike is the fiducial clause: below\n"
             "$p_\\mathrm{T}=10\\,$GeV$\\,\\times\\sin 15^\\circ=2.6\\,$GeV this sample puts\n"
             "cleanly reconstructed particles just outside\n"
             "$15^\\circ<\\theta<165^\\circ$ (median 12$^\\circ$, 10 GeV)",
             xy=(1.55, 70), xytext=(0.108, 62), fontsize=7.5,
             ha="left", va="center", color=COL["ink"],
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#C9D0DE", alpha=0.95),
             arrowprops=dict(arrowstyle="->", lw=0.9, color=COL["ref"]))
axf.set_ylabel("Fake rate [%]")
axf.set_ylim(0, 100)
axf.yaxis.set_major_locator(MultipleLocator(10))
axf.legend(loc="upper right", fontsize=7.5, framealpha=0.95)

axn.set_ylabel("Fake tracks / event")
axn.set_ylim(0, 1.3)
axn.yaxis.set_major_locator(MultipleLocator(0.25))
axn.set_xscale("log")
axn.set_xlim(0.095, 11)
axn.set_xlabel("$p_\\mathrm{T}$ [GeV]")

axm.text(0.03, 0.06, "$15^\\circ<\\theta<165^\\circ$,  $N_\\mathrm{hits}>10$",
         transform=axm.transAxes, fontsize=8.5, va="bottom",
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#C9D0DE", alpha=0.9))

# ---- set-level numbers for every variant, written on the figure -------------
hdr2 = (f"{'variant':<28}{'match':>7}{'hit eff':>9}{'cand/ev':>9}{'clones/ev':>11}"
        f"{'fake rate':>11}{'+fiducial':>11}{'fakes/ev':>10}")
hdr1 = " " * (28 + 7 + 9 + 9 + 11) + f"{'(no pure particle)':>22}"
lines = [hdr1, hdr2, "-" * len(hdr2)]
for label, s in rows:
    plain = label.replace("$\\geq$", ">=")
    def fmt(v, unit="%", w=11):
        return f"{'n/a':>{w}}" if v is None else f"{(100 * v if unit else v):{w - 1}.1f}{unit}"
    lines.append(f"{plain:<28}{100 * s['match']:6.1f}%{100 * s['hiteff']:8.1f}%"
                 f"{s['cand']:9.1f}{fmt(s['clones'], '', 11)}"
                 f"{fmt(s['fake_p'])}{fmt(s['fake_f'])}{fmt(s['fakes'], '', 10)}")
lines.append("")
lines.append("row 1 = the like-for-like GGTF definition (no candidate cut);  "
             "row 4 = recommended recipe")
lines.append("clones = extra >=75%-pure candidates for a particle another candidate "
             "already reconstructs")
fig.text(0.055, 0.180, "\n".join(lines), fontsize=7.4, family="monospace",
         va="bottom", ha="left", color=COL["ink"])

fig.text(0.055, 0.008,
         "Fake rate (panels 3-4 and the 'fake rate' column): fraction of emitted candidates with no true particle contributing\n"
         "$\\geq75\\%$ of their hits, i.e. candidates that do not correspond to a single trajectory. The '+fiducial' column adds our\n"
         "stricter clause, which also counts clean reconstructions of particles failing the IDEA selection ($\\theta$ outside\n"
         "$15^\\circ$-$165^\\circ$, or $\\leq10$ hits); on this sample that clause is 42-77% of all fakes and is the entire 1-2.7 GeV spike.\n"
         "Fragment merging: reconstruction step, no truth used; clusters whose condensation points lie within 0.15 in the learned\n"
         "embedding are merged into one track.\n"
         "$\\geq$4 hits: only clusters with at least 4 hits are emitted as track candidates (4 points is the minimum for a helix fit). It\n"
         "removes ~92% of the candidates but only ~75% of the impure ones, so the fake RATE rises while the number of fake\n"
         "tracks per event falls ~4x. Fake rate and fake count rank the variants differently; both are shown.\n"
         "Oracle merge: clusters of the same true particle fused using MC truth; upper bound for any merging algorithm, not a\n"
         "reconstruction step. Its cluster population is truth-derived, so no fake or clone numbers are quoted for it.",
         fontsize=8.0, va="bottom", ha="left", color=COL["ink"])
fig.subplots_adjust(bottom=0.335)
for e in ("png", "pdf"):
    fig.savefig(f"{OUT}/circe_loopers_eff_vs_pt.{e}", dpi=200, bbox_inches="tight")
print("wrote", f"{OUT}/circe_loopers_eff_vs_pt.png")
for line in lines:
    print(line)
