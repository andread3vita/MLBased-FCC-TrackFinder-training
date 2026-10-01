"""Operating-point frontier against the clone-sensitive metric.

The earlier frontier plotted efficiency against our own fake rate, which is
blind to clones: it keys off a candidate's majority particle, and a clone has a
perfectly good majority particle. Optimising against it chose a point that
merges aggressively, and merging turns out to make GGTF's clone-sensitive rate
worse, so the point was tuned against the wrong axis.

This re-plots with their rate -- unassigned / assigned above a hit cut, computed
from the real assignment -- on the fake axis, and marks their published 8% for
scale.

    PYTHONPATH=. python src/eval/plot_op_frontier.py --cfg consol_r3_nokeepall
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/marko.cechovic/cgatr-paper/literature/cgatr_fcc_pkg/eval_results"
GGTF_REFERENCE = 8.0  # per cent, De Vita's 2026 IDEA slide, N_hits > 10


def collect(cfg, subdirs):
    rows = []
    for sub in subdirs:
        for path in sorted(glob.glob(f"{BASE}/{cfg}/{sub}/*/summary.json")):
            with open(path) as f:
                s = json.load(f)
            if "ggtf_fake_rate_gt10" not in s:
                continue  # written before the exact assignment existed
            rows.append({
                "name": os.path.basename(os.path.dirname(path)),
                "group": sub,
                "tbeta": s.get("tbeta"),
                "td": s.get("td"),
                "merge_td": s.get("merge_td", 0.0),
                "attach_td": s.get("attach_td", 0.0),
                "min_hits": s.get("min_cluster_hits", 0),
                "helix": s.get("helix_tol", 0.0),
                "cand_per_ev": s.get("candidates_per_event", float("nan")),
                "ggtf10": 100 * s["ggtf_fake_rate_gt10"],
                "ggtf3": 100 * s["ggtf_fake_rate_gt3"],
                "ghost": 100 * s.get("ghost_rate", float("nan")),
                "clone": 100 * s.get("clone_rate", float("nan")),
                "def1": 100 * s["idea"]["match_rate"],
                "def2": 100 * s["idea"]["def2"],
                "n_events": s.get("n_events", 0),
            })
    return rows


def pareto(rows, xkey, ykey):
    """Points not dominated on (low x, high y)."""
    out = []
    for r in rows:
        if not any(o[xkey] <= r[xkey] and o[ykey] >= r[ykey]
                   and (o[xkey], -o[ykey]) != (r[xkey], -r[ykey]) for o in rows):
            out.append(r)
    return sorted(out, key=lambda r: r[xkey])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", default="consol_r3_nokeepall")
    ap.add_argument("--subdirs", default="op_resweep,helix_sweep")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    subdirs = args.subdirs.split(",")
    rows = collect(args.cfg, subdirs)
    if not rows:
        raise SystemExit(f"no summaries with the GGTF metric under "
                         f"{BASE}/{args.cfg}/{{{args.subdirs}}}")

    rows.sort(key=lambda r: r["ggtf10"])
    print(f"{'point':<40}{'cand/ev':>9}{'GGTF>10':>9}{'GGTF>3':>9}"
          f"{'ghost':>8}{'clone':>8}{'def1':>8}{'def2':>8}")
    print("-" * 99)
    for r in rows:
        print(f"{r['name']:<40}{r['cand_per_ev']:9.1f}{r['ggtf10']:8.1f}%"
              f"{r['ggtf3']:8.1f}%{r['ghost']:7.1f}%{r['clone']:7.1f}%"
              f"{r['def1']:7.1f}%{r['def2']:7.1f}%")

    front2 = pareto(rows, "ggtf10", "def2")
    print(f"\nPareto front on (GGTF>10, def2), {len(front2)} points:")
    for r in front2:
        print(f"  {r['name']:<40} GGTF>10={r['ggtf10']:6.1f}%  "
              f"def2={r['def2']:5.1f}%  def1={r['def1']:5.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))
    for ax, ykey, ylabel in (
        (axes[0], "def2", "IDEA efficiency, def2 [%]"),
        (axes[1], "def1", "IDEA match rate, def1 [%]"),
    ):
        for sub, marker in zip(subdirs, ("o", "s", "^", "D")):
            g = [r for r in rows if r["group"] == sub]
            if not g:
                continue
            helix = [r for r in g if r["helix"] > 0]
            plain = [r for r in g if r["helix"] == 0]
            if plain:
                ax.scatter([r["ggtf10"] for r in plain], [r[ykey] for r in plain],
                           s=34, marker=marker, alpha=0.75, color="#4878a8",
                           label=f"{sub}, no helix merge")
            if helix:
                ax.scatter([r["ggtf10"] for r in helix], [r[ykey] for r in helix],
                           s=44, marker=marker, alpha=0.85, color="#c04a3c",
                           label=f"{sub}, helix merge")
        front = pareto(rows, "ggtf10", ykey)
        ax.plot([r["ggtf10"] for r in front], [r[ykey] for r in front],
                color="0.35", lw=1.1, ls="--", zorder=1, label="Pareto front")
        ax.axvline(GGTF_REFERENCE, color="#2e7d32", lw=1.2)
        ax.set_xscale("log")
        ax.set_xlabel("GGTF fake rate, $N_{hits}>10$ [%]  (unassigned / assigned)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, which="both")

    axes[0].annotate("GGTF published 8%", xy=(GGTF_REFERENCE, 0.02),
                     xycoords=("data", "axes fraction"),
                     xytext=(0.30, 0.08), textcoords="axes fraction",
                     fontsize=9, color="#2e7d32",
                     arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.0))
    axes[0].legend(fontsize=8, loc="lower right", framealpha=0.92)

    fig.suptitle("Operating point against the clone-sensitive fake rate "
                 "(their metric, their assignment)", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.text(0.5, 0.005,
             "Their rate charges for clones because the assignment is "
             "one-to-one; ours does not, which is why the earlier frontier "
             "preferred aggressive merging.",
             ha="center", fontsize=8, color="0.35")

    out = args.out or f"{BASE}/{args.cfg}/op_frontier.png"
    fig.savefig(out, dpi=160)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
