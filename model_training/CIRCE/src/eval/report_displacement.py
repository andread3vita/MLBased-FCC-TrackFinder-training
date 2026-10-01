"""Tracking performance against track displacement, on both of the axes people mean by it.

Two different quantities go by "displacement" and they answer different questions:

  vertex_r  where the particle was *born*: sqrt(vx^2 + vy^2) of its MC production vertex.
            Prompt particles sit at ~0; b/c hadrons decay at tenths of a mm to a few mm,
            K_S and Lambda at centimetres, conversions and nuclear interactions wherever
            the material is. This is a truth-level property of the particle.

  d0        how far the track's own trajectory misses the beamline: the transverse impact
            parameter, the signed distance of closest approach of the helix to (0,0). This
            is the variable heavy-flavour tagging actually cuts on, because it is what a
            detector can measure. A particle born far out but heading outward can still
            have small d0, so the two axes are related but not interchangeable.

d0 is computed from the helix rather than read from a column: with the field along z, a track
of transverse momentum pt and charge q curves with radius R = pt / (0.3 B), its circle centre
sits at `vertex + q R (sin phi, -cos phi)`, and d0 = ||centre|| - R. B is not a constant anywhere
in this codebase, so it is measured from the data instead -- an algebraic circle fit to the hits
of 4000 prompt, >40-hit tracks gives 2.002 T (IQR 2.000-2.005), i.e. IDEA's design 2 T.

Why a learned tracker is asked this: conventional pattern recognition is usually seeded on the
assumption that a track points back to the interaction point, so it degrades with displacement.
A model that clusters hits carries no such prior, so a flat curve is a claim worth making.

METRIC. Both of GGTF's efficiency definitions are reported, because the two disagree and each is
the headline somewhere:
  def1 (match rate) matched to a cluster of > 75% purity -- the paper draft's vertex-R figure
  def2             purity > 50% AND hit efficiency > 50% -- GGTF's own IDEA plots
Their fake rate deliberately does not appear. It is a per-*cluster* quantity, and a fake cluster
corresponds to no true particle, so it has no production vertex and cannot be binned on either
axis. Efficiency is the only side of the ledger this plot can carry.

THE CONFOUND, which the uncontrolled version of this plot walks straight into: the median target
falls from ~120 signal hits in the beamspot to a handful beyond a metre, because a particle
created inside the tracking volume has less volume left to cross. Binned naively the curve
measures track length and calls it displacement. `--min_hits` holds length roughly fixed.

A naming trap not inherited here: the `displaced` row in summary.json comes from
`is_reconstructable_displaced`, a kinematic acceptance cut (pt > 1 GeV, 10 < theta < 170 deg,
charged) with no displacement requirement in it at all. It is named after the study it came from
and it is not this measurement.

    PYTHONPATH=. python src/eval/report_displacement.py --preset paper
    PYTHONPATH=. python src/eval/report_displacement.py --preset algebra
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import polars as pl

B_TESLA = 2.002  # measured from track curvature, see module docstring

REGIME = "eval_results/ggtf_regime"
OP = "tb0.6_td0.2_mh4_hmh4_hx0.15_mth3"

# The full-dataset run behind every \prelim number in sn-article.tex: seeds 1-180, epoch 19.
PAPER_CACHE = "eval_results/ep19_nokeepall/fcc_unmerged/cache.parquet"

# Labelled by the sample rather than by how much of it was used. sn-article.tex calls this
# "$e^+e^-\to Z\to q\bar q$ events with $q\in\{u,d,s\}$" for the IDEA detector, so the figure
# says that and the seed range stays in the caption where it belongs.
PRESETS = {
    "paper": [(r"CIRCE, IDEA $e^+e^-\!\to Z\to q\bar{q}$ (uds)", PAPER_CACHE)],
    "algebra": [
        (a, os.path.join(REGIME, a, OP, "cache.parquet"))
        for a in ("projective_parity", "conformal_parity", "conformal_wiredir")
    ],
}

AXES = {
    "vertex_r": ([0.0, 0.1, 1.0, 10.0, 50.0, 350.0, 1000.0, 1e9],
                 ["<0.1", "0.1-1", "1-10", "10-50", "50-350", "350-1000", ">1000"],
                 "production-vertex radius [mm]"),
    "d0":       ([0.0, 0.05, 0.2, 1.0, 5.0, 20.0, 100.0, 1e9],
                 ["<0.05", "0.05-0.2", "0.2-1", "1-5", "5-20", "20-100", ">100"],
                 "transverse impact parameter |d0| [mm]"),
}


def add_d0(df: pl.DataFrame) -> pl.DataFrame:
    """|d0| from the helix, in mm. Zero for a prompt track by construction."""
    r_mm = (pl.col("pt") / (0.299792458 * B_TESLA)) * 1000.0
    q = pl.col("charge").sign()
    cx = pl.col("vx") + q * r_mm * pl.col("phi").sin()
    cy = pl.col("vy") - q * r_mm * pl.col("phi").cos()
    return df.with_columns(
        ((cx ** 2 + cy ** 2).sqrt() - r_mm.abs()).abs().alias("d0"))


def curve(df: pl.DataFrame, axis: str, min_hits: int) -> list[dict]:
    edges, labels, _ = AXES[axis]
    x = df[axis].to_numpy()
    p = df["purity_of_match"].to_numpy()
    e = df["efficiency_per_hit"].to_numpy()
    h = df["n_hits_signal"].to_numpy()
    out = []
    for i, lab in enumerate(labels):
        m = (x >= edges[i]) & (x < edges[i + 1]) & (h >= min_hits)
        n = int(m.sum())
        out.append({
            "bin": lab, "n": n,
            "med_hits": float(np.median(h[m])) if n else float("nan"),
            "def1": float((p[m] > 0.75).mean()) if n else float("nan"),
            "def2": float(((p[m] > 0.5) & (e[m] > 0.5)).mean()) if n else float("nan"),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default="paper")
    ap.add_argument("--min_hits", type=int, default=20,
                    help="floor on signal hits per target, to hold track length fixed")
    ap.add_argument("--idea_cut", action="store_true", default=True,
                    help="restrict to IDEA-reconstructable particles, as the draft does")
    ap.add_argument("--metrics", choices=["both", "def1", "def2"], default="both",
                    help="which efficiency definitions to draw; tables always show both")
    ap.add_argument("--outdir", default="eval_results/displacement")
    args = ap.parse_args()

    print(__doc__)
    data = {}
    for name, path in PRESETS[args.preset]:
        if not os.path.exists(path):
            print(f"  [skip] {name}: no cache at {path}")
            continue
        df = pl.read_parquet(path)
        if args.idea_cut and "is_reconstructable_idea" in df.columns:
            df = df.filter(pl.col("is_reconstructable_idea"))
        data[name] = add_d0(df)

    if not data:
        print("nothing to report")
        return

    os.makedirs(args.outdir, exist_ok=True)
    for axis in AXES:
        _, labels, xlabel = AXES[axis]
        raw = {k: curve(v, axis, 0) for k, v in data.items()}
        ctl = {k: curve(v, axis, args.min_hits) for k, v in data.items()}

        for curves, title in (
                (raw, f"{xlabel} -- UNCONTROLLED (see median hits; do not quote)"),
                (ctl, f"{xlabel} -- length-controlled at >= {args.min_hits} hits")):
            first = next(iter(curves.values()))
            print(f"\n{title}\n")
            hdr = (f"{'bin':>12}{'targets':>9}{'med hits':>10}"
                   + "".join(f"{n[:26]:>28}" for n in curves))
            print(hdr)
            print("-" * len(hdr))
            for i, lab in enumerate(labels):
                row = f"{lab:>12}{first[i]['n']:9d}{first[i]['med_hits']:10.0f}"
                for name in curves:
                    c = curves[name][i]
                    row += (f"{c['def1'] * 100:15.1f}%{c['def2'] * 100:11.1f}%"
                            if c["n"] else f"{'--':>28}")
                print(row)
            print(f"{'':31}" + "".join(f"{'def1':>16}{'def2':>12}" for _ in curves))

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nmatplotlib not available, tables only")
            return

        centres = np.arange(len(labels))
        fig, (ax, axn) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
        # With one definition drawn the legend only has to name the arm, and the y-axis carries
        # which efficiency it is; with both it has to say per curve.
        show = {"both": ("def1", "def2"), "def1": ("def1",), "def2": ("def2",)}[args.metrics]
        styles = {"def1": ("o", "-", "match rate (>75% purity)"),
                  "def2": ("s", "--", "def2 (>50% purity & >50% hit eff.)")}
        for name in ctl:
            for key in show:
                marker, ls, desc = styles[key]
                ax.plot(centres, [c[key] * 100 for c in ctl[name]], marker=marker, ls=ls,
                        label=name if len(show) == 1 else f"{name} — {desc}")
        ax.set_ylabel(styles[show[0]][2] + " [%]" if len(show) == 1 else "efficiency [%]")
        ax.set_title("Tracking efficiency against track displacement\n"
                     f"IDEA-reconstructable, targets with >= {args.min_hits} signal hits "
                     "so track length is held fixed")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        axn.bar(centres, [c["n"] for c in next(iter(ctl.values()))], color="0.7")
        axn.set_yscale("log")
        axn.set_ylabel("targets")
        axn.set_xlabel(xlabel)
        axn.set_xticks(centres)
        axn.set_xticklabels(labels, rotation=30, ha="right")
        axn.grid(alpha=0.3)
        fig.tight_layout()
        suffix = "" if args.metrics == "both" else f"_{args.metrics}"
        out = os.path.join(args.outdir, f"{args.preset}{suffix}_efficiency_vs_{axis}.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
