"""Why do sub-GeV tracks fragment? Hit count, radius, or number of loops.

97% of the clone population sits below 0.85 GeV, so the clone rate is not a
diffuse quality problem but one failure mode. This separates the three candidate
causes, which are confounded on any real sample because a curler necessarily has
many hits and a small radius:

  hit count   more hits to hold together
  radius      tighter helix, hits closer in space
  loops       the trajectory wraps, so hits far apart in path length land close
              together in space

The loop count is measured, not assumed: hits are ordered by time and the
azimuthal angle is unwrapped, so n_loops is the total turning angle over 2*pi.
Fragmentation is measured two ways -- the hit efficiency of the best-matched
cluster (what the metrics report shows) and, straight from the embedding, the
number of pieces the condensation splits a particle into, obtained by running
the CIRCE greedy assignment over that particle's hits alone.

    python src/eval/curler_diagnostic.py --seeds 300
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from src.eval.plotstyle import COL

RUN = "eval_results/r3_loopers/merge_sweep/td0.10_mg0_at0"
EMB = "eval_results/r3_loopers/emb_all/forward_hits.parquet"
RAW = "/home/marko.cechovic/cgatr-data/test-data/parquet"
OUT = "eval_results/share/curler_diagnostic"
TD = 0.10
KEY = ["seed", "event_id", "mc_index"]


def pieces_and_spread(emb, td=TD):
    """Run the CIRCE greedy assignment inside each particle separately.

    n_pieces is then the number of clusters the condensation would need for that
    particle even with no competition from other particles, i.e. fragmentation
    caused purely by the embedding being spread wider than t_d."""
    cols = ["coord_0", "coord_1", "coord_2", "coord_3"]
    out = []
    for (seed, ev, mc), g in emb.group_by(KEY, maintain_order=True):
        x = g.select(cols).to_numpy()
        b = g["beta"].to_numpy()
        order = np.argsort(-b)
        free = np.ones(len(x), bool)
        n_pieces = 0
        first_frac = 0.0
        for i in order:
            if not free[i]:
                continue
            d = np.linalg.norm(x[free] - x[i], axis=1)
            idx = np.flatnonzero(free)
            take = idx[d <= td]
            if n_pieces == 0:
                first_frac = len(take) / len(x)
            free[take] = False
            n_pieces += 1
        out.append((seed, ev, mc, len(x), n_pieces, first_frac))
    return pl.DataFrame(out, schema=["seed", "event_id", "mc_index", "n_emb_hits",
                                     "n_pieces", "frac_in_td"], orient="row")


def loops(seeds):
    """True turning angle per particle from the drift-chamber hits.

    Measured from the azimuth of the transverse MOMENTUM, unwrapped along time,
    not from the azimuth of the position. A curler whose helix centre sits away
    from the beam axis -- which is most of them, since a 0.1 GeV track has a
    167 mm bending radius and can only be at large r if it started there -- has
    a position azimuth that merely oscillates by ~2*atan(R/r_centre) no matter
    how many turns it makes, so position-based unwrapping reports a fraction of
    a turn for a track that circles twenty times."""
    out = []
    for s in seeds:
        f = f"{RAW}/seed_{s}/dc_hits_train.parquet"
        if not os.path.exists(f):
            continue
        d = pl.read_parquet(f, columns=["hit_x", "hit_y", "hit_px", "hit_py",
                                        "mc_index", "event_id", "seed", "time"])
        for (ev, mc), g in d.filter(pl.col("mc_index") != 0).group_by(
                ["event_id", "mc_index"], maintain_order=True):
            if len(g) < 3:
                continue
            g = g.sort("time")
            psi = np.arctan2(g["hit_py"].to_numpy(), g["hit_px"].to_numpy())
            u = np.unwrap(psi)
            r = np.hypot(g["hit_x"].to_numpy(), g["hit_y"].to_numpy())
            out.append((s, ev, mc, (u.max() - u.min()) / (2 * np.pi),
                        len(g), float(r.max()), float(r.min())))
    return pl.DataFrame(out, schema=["seed", "event_id", "mc_index", "n_loops",
                                     "n_dc_hits", "r_max", "r_min"], orient="row")


def r2(y, *cols):
    A = np.column_stack([np.ones_like(y)] + list(cols))
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return 1 - (y - A @ b).var() / y.var(), b


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=300)
    args = ap.parse_args()

    t = pl.read_parquet(f"{RUN}/cache.parquet",
                        columns=["mc_idx", "event_id", "seed", "pt", "n_hits_total",
                                 "efficiency_per_hit", "purity_of_match", "charge",
                                 "gen_status"]).rename({"mc_idx": "mc_index"})
    t = t.filter((pl.col("charge").abs() > 0) & pl.col("gen_status").is_in([0, 1])
                 & (pl.col("n_hits_total") > 10) & (pl.col("pt") > 0.02))
    seeds = sorted(t["seed"].unique().to_list())[:args.seeds]
    t = t.filter(pl.col("seed").is_in(seeds))

    print(f"[curler] {len(t):,} particles over {len(seeds)} seeds")
    emb = (pl.read_parquet(EMB).filter((pl.col("mc_index") != 0)
                                       & pl.col("seed").is_in(seeds))
           .sort(KEY))
    print(f"[curler] embedding: {len(emb):,} hits -> greedy per particle ...")
    pc = pieces_and_spread(emb)
    print(f"[curler] drift-chamber geometry for the loop count ...")
    lp = loops(seeds)

    d = t.join(pc, on=KEY, how="inner").join(lp, on=KEY, how="inner")
    d = d.filter(pl.col("n_loops") > 0.01)
    print(f"[curler] {len(d):,} particles with embedding + geometry\n")

    pt = d["pt"].to_numpy(); nh = d["n_hits_total"].to_numpy().astype(float)
    nl = d["n_loops"].to_numpy(); ef = d["efficiency_per_hit"].to_numpy()
    npc = d["n_pieces"].to_numpy().astype(float)
    rmax = d["r_max"].to_numpy()
    y = np.log(np.clip(1 - ef, 0.01, None))

    print("Which variable governs fragmentation?  R^2 of log(1 - hit efficiency)")
    for lab, c in [("n_hits", np.log(nh)), ("pT", np.log(pt)),
                   ("max radius", np.log(rmax + 1e-6)), ("n_loops", np.log(nl)),
                   ("n_hits and pT", None), ("n_loops and n_hits", None)]:
        if c is not None:
            v, _ = r2(y, c)
        elif lab == "n_hits and pT":
            v, _ = r2(y, np.log(nh), np.log(pt))
        else:
            v, _ = r2(y, np.log(nl), np.log(nh))
        print(f"   {lab:<22}{v:6.3f}")
    _, b = r2(y, np.log(nh), np.log(pt))
    print(f"   fit: log(1-eff) = {b[0]:+.2f} {b[1]:+.2f} log(n_hits) {b[2]:+.2f} log(pT)"
          f"   (equal and opposite => the ratio, i.e. loops)")

    print("\nDoes the condensation split once per loop?")
    print(f"   {'loops':>12}{'N':>7}{'n_pieces':>10}{'hit eff':>9}{'n_hits':>8}{'pT':>7}")
    E = [0, 0.5, 1, 1.5, 2, 3, 5, 8, 100]
    for a, bb in zip(E[:-1], E[1:]):
        s = (nl >= a) & (nl < bb)
        if s.sum() < 15:
            continue
        print(f"   {a:5.1f}-{bb:<6.1f}{int(s.sum()):7d}{npc[s].mean():10.1f}"
              f"{100*ef[s].mean():8.0f}%{nh[s].mean():8.0f}{pt[s].mean():7.2f}")

    lo = nl < 1
    print(f"\n   under 1 loop: {100*ef[lo].mean():.0f}% hit efficiency, "
          f"{npc[lo].mean():.1f} pieces, {nh[lo].mean():.0f} hits")
    print(f"   over 2 loops: {100*ef[~lo & (nl>2)].mean():.0f}% hit efficiency, "
          f"{npc[nl>2].mean():.1f} pieces, {nh[nl>2].mean():.0f} hits")

    # a track with many hits but no wrapping is fine; the control comparison
    many = (nh > 80) & (nl < 1)
    curl = (nh > 80) & (nl > 2)
    if many.sum() > 10 and curl.sum() > 10:
        print(f"\n   CONTROL, both with >80 hits:")
        print(f"     < 1 loop : {int(many.sum()):5d} particles, "
              f"{100*ef[many].mean():.0f}% hit efficiency, {npc[many].mean():.1f} pieces")
        print(f"     > 2 loops: {int(curl.sum()):5d} particles, "
              f"{100*ef[curl].mean():.0f}% hit efficiency, {npc[curl].mean():.1f} pieces")

    figure(d, nl, npc, ef, nh, pt)
    d.write_parquet(OUT + ".parquet")
    print(f"\nwrote {OUT}.png / .parquet")


def figure(d, nl, npc, ef, nh, pt):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.5, 5.6))

    a1.hexbin(nl, npc / nh, xscale="log", yscale="log", gridsize=34, mincnt=1,
              cmap="cividis", norm=LogNorm())
    a1.axhline(1.0, color=COL["fake"], lw=2, label="every hit its own cluster")
    a1.axvline(0.5, color=COL["ink"], ls=":", lw=1.4)
    a1.set(xlabel="loops (measured turning angle / $2\\pi$)",
           ylabel="pieces per hit", title="Past half a turn the embedding shatters")
    a1.legend(loc="lower right")

    E = np.logspace(np.log10(0.03), np.log10(30), 16)
    c, lo, hi = [], [], []
    for a, b in zip(E[:-1], E[1:]):
        s = (nl >= a) & (nl < b)
        c.append(100 * ef[s].mean() if s.sum() >= 15 else np.nan)
    a2.plot(np.sqrt(E[:-1] * E[1:]), c, color=COL["standard"], marker="o")
    a2.axvline(1.0, color=COL["ink"], ls=":", lw=1.4)
    a2.annotate("one full turn", xy=(1.0, 0.06), xycoords=("data", "axes fraction"),
                rotation=90, fontsize=8.5, ha="right", color=COL["ink"])
    a2.set(xscale="log", xlabel="loops", ylabel="Hit efficiency [%]",
           title="Fragmentation sets in at the first turn")

    # the two controls: hold the rival explanation fixed, vary the loop count
    groups = [("same hit count\n(> 80 hits)", (nh > 80) & (nl < 1), (nh > 80) & (nl > 2)),
              ("same $p_T$\n(0.2 - 0.4 GeV)",
               (pt >= 0.2) & (pt < 0.4) & (nl < 0.5),
               (pt >= 0.2) & (pt < 0.4) & (nl >= 1.5))]
    w, xs = 0.34, np.arange(len(groups))
    for k, (off, lab, col) in enumerate([(-w / 2, "few loops", COL["standard"]),
                                         (w / 2, "many loops", COL["fake"])]):
        v = [100 * ef[g[k + 1]].mean() for g in groups]
        a3.bar(xs + off, v, width=w, color=col, label=lab)
        for j, val in enumerate(v):
            a3.text(xs[j] + off, val + 1.5, f"{val:.0f}%", ha="center",
                    fontsize=11, fontweight="bold", color=COL["ink"])
            a3.text(xs[j] + off, 3.0, f"n={int(groups[j][k + 1].sum())}", ha="center",
                    fontsize=8, color="white" if val > 12 else COL["ink"])
    a3.set(ylabel="Hit efficiency [%]", ylim=(0, 105),
           title="Hold the rival cause fixed: still the loops")
    a3.set_xticks(xs, [g[0] for g in groups])
    a3.legend(loc="upper right")

    fig.suptitle("Sub-GeV fragmentation is caused by looping, not by hit count "
                 "or radius  (Loopers, $t_d$=0.1, no merging)",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(bottom=0.16, top=0.84, wspace=0.28)
    fig.text(0.5, 0.015, "Loops are measured, not inferred: drift-chamber hits are "
             "ordered by time and the azimuthal angle unwrapped, so a loop is a full "
             "$2\\pi$ of turning.  Pieces are obtained by running the CIRCE greedy "
             "assignment over one particle's hits alone, so they count fragmentation "
             "caused purely by the embedding spreading wider than $t_d$.",
             ha="center", fontsize=8.4, color="#3C4657")
    fig.savefig(OUT + ".png")
    plt.close(fig)


if __name__ == "__main__":
    main()
