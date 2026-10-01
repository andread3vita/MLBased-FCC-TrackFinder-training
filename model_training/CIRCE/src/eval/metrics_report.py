"""Canonical metric report for a CIRCE evaluation run: JSON + markdown + figure.

This is the single entry point for "what does this checkpoint do". Point it at
one or more finished eval directories (any dir holding cache.parquet,
cache_clusters.parquet and summary.json) and it writes <out>.json, <out>.md and
<out>.png. It reads only the cached parquets, so it needs no GPU and no model.

    python src/eval/metrics_report.py --out eval_results/share/final_zqq \\
        --title "Zqq, seeds 181-200" \\
        "CIRCE (t_d=0.2)=eval_results/consol_r3_nokeepall/fcc_unmerged" \\
        "adopted=eval_results/consol_r3_nokeepall/merge_sweep/td0.10_mg0.20_at0_mh4"

FROZEN DEFINITIONS (settled 2026-07-30, see paper_adjustment_candidates/README.md)

Efficiency, over particles passing the IDEA reconstructable selection
(pt > 0.1 GeV, 15 < theta < 165 deg, > 10 hits, charged, primary):
  match rate         purity of the best-matched cluster > 75%      (GGTF def 1)
  strict match rate  the above AND that cluster holds > 50% of the particle's
                     hits, so a pure fragment does not count as a track
  def2               purity > 50% AND hit efficiency > 50%         (GGTF def 2)
  hit efficiency     mean fraction of a particle's hits in its best cluster

Candidate quality, over every emitted cluster:
  ghost rate         purity < 75%: the cluster does not correspond to a single
                     particle. THIS IS THE FAKE RATE. The reconstructable
                     selection defines the efficiency denominator and appears
                     nowhere here -- a clean reconstruction of a particle we did
                     not ask for is not a fake, and charging it as one uses the
                     same cut twice (the model's loss never sees pt or theta).
  clone rate         extra >75%-pure clusters for a particle another cluster
                     already reconstructs
  purity             mean cluster purity

Never quote the ghost rate alone. It is a fraction of emitted clusters and
clones are pure by construction, so fragmenting more pushes it down: at
t_d=0.2 the model scores 5.7% only because 87% of its 609 clusters/event are
duplicates. candidates_per_event and ghosts_per_distinct (ghosts over ghosts +
distinct particles found) are reported beside it for exactly this reason.

  legacy_fake_rate   the old definition (ghost OR majority particle outside the
                     reconstructable selection), carried for continuity with
                     figures made before 2026-07-30. Do not quote it.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from src.eval.fcc_cache_parallel import candidate_stats, overall
from src.eval.plotstyle import COL

EDGES = np.logspace(np.log10(0.05), np.log10(10.0), 15)
MIN_N = 40
SERIES_COL = [COL["standard"], COL["keepall"], COL["eff"], COL["fake"], COL["ink"]]
SERIES_MRK = ["o", "D", "s", "^", "v"]


def wilson(k, n):
    """68% Wilson interval; safer than sqrt(k)/n for rates near 0 or 1."""
    p = k / n
    z2 = 1.0
    denom = 1 + z2 / n
    c = (p + z2 / (2 * n)) / denom
    half = np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return p, max(0.0, p - (c - half)), max(0.0, (c + half) - p)


def load(d):
    s = json.load(open(os.path.join(d, "summary.json")))
    t = pl.read_parquet(os.path.join(d, "cache.parquet"))
    cp = os.path.join(d, "cache_clusters.parquet")
    c = pl.read_parquet(cp) if os.path.exists(cp) else None
    return s, t, c


def set_level(d):
    s, t, c = load(d)
    n_events = s.get("n_events") or len(t.select(["event_id", "seed"]).unique())
    r = t.filter(pl.col("is_reconstructable_idea"))
    p = r["purity_of_match"].to_numpy()
    e = r["efficiency_per_hit"].to_numpy()

    m = {"n_events": n_events, "n_reconstructable": len(r),
         "reconstructable_per_event": len(r) / n_events}
    m.update({k: v for k, v in overall(t, "is_reconstructable_idea").items()
              if k in ("efficiency", "match_rate", "def2", "split", "multiple", "bad")})
    m["strict_match_rate"] = float(((p > 0.75) & (e > 0.5)).mean())

    if c is None:                      # oracle merge: truth-derived, no fakes
        m["truth_derived"] = True
        return m
    m.update(candidate_stats(c, n_events))
    m["purity"] = float(c["purity"].mean())
    m["legacy_fake_rate"] = float(c["is_fake_idea"].mean())
    good = c.filter(pl.col("purity") >= 0.75)
    n_uniq = len(good.group_by(["matched_mc_idx", "event_id", "seed"]).len())
    n_ghost = int((c["purity"] < 0.75).sum())
    m["distinct_per_event"] = n_uniq / n_events
    m["ghosts_per_distinct"] = n_ghost / max(n_ghost + n_uniq, 1)
    return m


def binned(d):
    """Per-pT curves. Particle panels are binned by true pt over the
    reconstructable set; cluster panels by the majority particle's pt, so
    clusters with no particle majority reach the set-level numbers only."""
    _, t, c = load(d)
    r = t.filter(pl.col("is_reconstructable_idea"))
    pt = r["pt"].to_numpy()
    pur = r["purity_of_match"].to_numpy()
    eff = r["efficiency_per_hit"].to_numpy()
    matched = r["matched"].to_numpy().astype(float)
    strict = ((pur > 0.75) & (eff > 0.5)).astype(float)

    out = {k: [] for k in ("x", "match", "match_lo", "match_hi", "strict",
                           "hiteff", "hiteff_e")}
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        s = (pt >= a) & (pt < b)
        n = int(s.sum())
        if n < MIN_N:
            continue
        p, lo, hi = wilson(matched[s].sum(), n)
        out["x"].append(np.sqrt(a * b))
        out["match"].append(p); out["match_lo"].append(lo); out["match_hi"].append(hi)
        out["strict"].append(strict[s].mean())
        out["hiteff"].append(eff[s].mean())
        out["hiteff_e"].append(eff[s].std() / np.sqrt(n))

    if c is not None:
        cc = c.drop_nulls("pt").with_row_index("_i")
        cpt = cc["pt"].to_numpy()
        cpur = cc["purity"].to_numpy()
        ghost = (cpur < 0.75).astype(float)
        # a pure cluster is a clone if another pure cluster already claimed the
        # same particle; ranking within the pure subset only, first one wins
        pure = cc.filter(pl.col("purity") >= 0.75).with_columns(
            (pl.int_range(pl.len()).over(["matched_mc_idx", "event_id", "seed"]) > 0)
            .alias("_clone"))
        clone = np.zeros(len(cc))
        clone[pure["_i"].to_numpy()] = pure["_clone"].to_numpy()
        n_events = json.load(open(os.path.join(d, "summary.json")))["n_events"]
        for k in ("cx", "ghost", "ghost_lo", "ghost_hi", "purity", "clone",
                  "ghost_per_ev"):
            out[k] = []
        for a, b in zip(EDGES[:-1], EDGES[1:]):
            s = (cpt >= a) & (cpt < b)
            n = int(s.sum())
            if n < MIN_N:
                continue
            k_ghost = ghost[s].sum()
            p, lo, hi = wilson(k_ghost, n)
            out["cx"].append(np.sqrt(a * b))
            out["ghost"].append(p); out["ghost_lo"].append(lo); out["ghost_hi"].append(hi)
            out["purity"].append(cpur[s].mean())
            out["clone"].append(clone[s].mean())
            out["ghost_per_ev"].append(k_ghost / n_events)
    return {k: np.asarray(v, float) for k, v in out.items()}


def markdown(rows, title):
    h = ("| configuration | cand/ev | match | strict | def2 | hit eff | purity "
         "| ghost | clone | distinct/ev | ghost/distinct | legacy fake |")
    out = [f"### {title}", "", h, "|" + "---|" * 12]
    for lab, m in rows:
        if m.get("truth_derived"):
            out.append(f"| {lab} | - | {100*m['match_rate']:.1f}% | "
                       f"{100*m['strict_match_rate']:.1f}% | {100*m['def2']:.1f}% | "
                       f"{100*m['efficiency']:.1f}% | truth-derived |" + " - |" * 5)
            continue
        out.append(
            f"| {lab} | {m['candidates_per_event']:.1f} | {100*m['match_rate']:.1f}% "
            f"| {100*m['strict_match_rate']:.1f}% | {100*m['def2']:.1f}% "
            f"| {100*m['efficiency']:.1f}% | {100*m['purity']:.1f}% "
            f"| {100*m['ghost_rate']:.2f}% | {100*m['clone_rate']:.1f}% "
            f"| {m['distinct_per_event']:.1f} | {100*m['ghosts_per_distinct']:.1f}% "
            f"| {100*m['legacy_fake_rate']:.1f}% |")
    return "\n".join(out)


def figure(runs, rows, out_png, title):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))
    (a_m, a_e, a_p), (a_g, a_c, a_n) = axes
    for i, ((lab, d), (_, m)) in enumerate(zip(runs, rows)):
        b = binned(d)
        col, mk = SERIES_COL[i % 5], SERIES_MRK[i % 5]
        a_m.errorbar(b["x"], 100 * b["match"],
                     yerr=[100 * b["match_lo"], 100 * b["match_hi"]],
                     color=col, marker=mk, ls="-", capsize=2, label=lab)
        a_m.plot(b["x"], 100 * b["strict"], color=col, marker=mk, ls="--",
                 alpha=0.75, ms=4)
        a_e.errorbar(b["x"], 100 * b["hiteff"], yerr=100 * b["hiteff_e"],
                     color=col, marker=mk, capsize=2, label=lab)
        if "cx" not in b:
            continue
        a_p.plot(b["cx"], 100 * b["purity"], color=col, marker=mk, label=lab)
        a_g.errorbar(b["cx"], 100 * b["ghost"],
                     yerr=[100 * b["ghost_lo"], 100 * b["ghost_hi"]],
                     color=col, marker=mk, capsize=2, label=lab)
        a_c.plot(b["cx"], 100 * b["clone"], color=col, marker=mk, label=lab)
        a_n.plot(b["cx"], b["ghost_per_ev"], color=col, marker=mk, label=lab)

    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.set_xlabel(r"$p_T$ [GeV]")
        ax.axvspan(EDGES[0], 0.85, color=COL["grey"], alpha=0.16, lw=0, zorder=0)
    a_m.set(title="Track match rate", ylabel="Match rate [%]")
    a_m.set_ylim(top=101)
    a_m.yaxis.set_major_locator(MultipleLocator(5))
    a_e.set(title="Track hit efficiency", ylabel="Hit efficiency [%]")
    a_p.set(title="Cluster purity", ylabel="Mean purity [%]")
    a_g.set(title="Ghost rate  (= the fake rate)", ylabel="Ghost rate [%]")
    a_c.set(title="Clone rate", ylabel="Clones / cluster [%]")
    a_n.set(title="Ghosts per event", ylabel="Ghosts / event / bin")

    a_c.annotate("curlers: 97% of all clones\nsit below 0.85 GeV",
                 xy=(0.73, 0.30), xycoords="axes fraction", ha="center",
                 fontsize=8.6, color=COL["ink"])
    h, l = a_m.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=len(l), frameon=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    tbl = ["configuration".ljust(30) + "".join(
        s.rjust(w) for s, w in [("cand/ev", 9), ("match", 8), ("strict", 8),
                                ("hit eff", 9), ("purity", 8), ("ghost", 8),
                                ("clone", 8), ("distinct/ev", 13)])]
    for lab, m in rows:
        if m.get("truth_derived"):
            tbl.append(lab.ljust(30) + "truth-derived (oracle merge): "
                       f"match {100*m['match_rate']:.1f}%, "
                       f"strict {100*m['strict_match_rate']:.1f}%")
            continue
        tbl.append(lab.ljust(30) + "".join(s.rjust(w) for s, w in [
            (f"{m['candidates_per_event']:.1f}", 9),
            (f"{100*m['match_rate']:.1f}%", 8), (f"{100*m['strict_match_rate']:.1f}%", 8),
            (f"{100*m['efficiency']:.1f}%", 9), (f"{100*m['purity']:.1f}%", 8),
            (f"{100*m['ghost_rate']:.2f}%", 8), (f"{100*m['clone_rate']:.1f}%", 8),
            (f"{m['distinct_per_event']:.1f}", 13)]))
    fig.subplots_adjust(bottom=0.215, hspace=0.33, wspace=0.24, top=0.885)
    fig.text(0.5, 0.145, "\n".join(tbl), ha="center", va="top", family="monospace",
             fontsize=8.4, color=COL["ink"])
    fig.text(0.5, 0.028,
             "Solid: match rate (purity of the matched cluster > 75%, GGTF def 1).  "
             "Dashed: strict match rate (also > 50% of the particle's hits).  "
             "Efficiency panels are binned by true $p_T$ over IDEA-reconstructable "
             "particles;\ncluster panels by the majority particle's $p_T$.  The ghost "
             "rate is the fake rate: purity < 75%, i.e. the cluster is not a single "
             "particle.  It is a fraction of emitted clusters, so it falls when the\n"
             "model fragments more (clones are pure) -- read it together with "
             "candidates/event and the clone rate.  Error bars are 68% Wilson.  "
             "Shaded: the sub-0.85 GeV curling region.",
             ha="center", va="bottom", fontsize=8.4, color="#3C4657")
    fig.savefig(out_png)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", metavar="LABEL=DIR")
    ap.add_argument("--out", required=True, help="output path prefix")
    ap.add_argument("--title", default="CIRCE evaluation")
    args = ap.parse_args()

    runs = []
    for r in args.runs:
        lab, _, d = r.rpartition("=")   # last '=', so labels may contain one
        if not d or not os.path.exists(os.path.join(d, "summary.json")):
            sys.exit(f"no summary.json under {d!r} (expected LABEL=DIR)")
        runs.append((lab, d))

    rows = [(lab, set_level(d)) for lab, d in runs]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump({lab: m for lab, m in rows}, f, indent=2)
    md = markdown(rows, args.title)
    with open(args.out + ".md", "w") as f:
        f.write(md + "\n")
    figure(runs, rows, args.out + ".png", args.title)
    print(md)
    print(f"\nwrote {args.out}.json / .md / .png")


if __name__ == "__main__":
    main()
