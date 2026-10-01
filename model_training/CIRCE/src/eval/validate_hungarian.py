"""Is our greedy stand-in for GGTF's assignment the same as the real thing?

`ggtf_fake_rate.py` reconstructs their matching from the cached per-candidate
aggregates, which only record each candidate's majority particle. That makes it
a greedy rule -- per particle, keep the candidate with the largest IoU -- where
they run `scipy.optimize.linear_sum_assignment` over the whole cluster-by-
particle IoU matrix (`src/layers/inference_oc_tracks.py`, threshold 0.02).

Greedy and optimal can differ: greedy can hand a particle to a candidate that
the optimum would have spent elsewhere. This script re-clusters a sample of
events from a forward cache, builds the full IoU matrix per event, and compares

    (a) linear_sum_assignment on the gated matrix, exactly as they do it,
    (b) our greedy rule,

reporting the resulting fake counts side by side. The plan lists the emulation
as an open risk; this closes it or quantifies it.

    PYTHONPATH=. python src/eval/validate_hungarian.py \
        --cache_path eval_results/consol_r3_nokeepall/emb_all \
        --tbeta 0.1 --td 0.2 --n_events 400
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

from src.eval.fcc_cache_parallel import attach_debris, merge_fragments
from src.eval.ggtf_assign import IOU_THRESHOLD, contingency, iou_matrix
from src.eval_sweep_v33 import get_clustering_greedy
from src.lowpt_op_sweep import cache_from_dataframe


def assign(iou, inter, mode):
    """Per-candidate assignment flag under one of the two rules."""
    gated = np.where(iou < IOU_THRESHOLD, 0.0, iou)
    assigned = np.zeros(gated.shape[0], dtype=bool)
    if not gated.size:
        return assigned

    if mode == "hungarian":
        r, c = linear_sum_assignment(-gated)
        keep = gated[r, c] > 0
        assigned[r[keep]] = True
    else:
        # What the cached aggregates support: each candidate is grouped by its
        # MAJORITY particle (raw overlap, the stored matched_mc_idx), and within
        # a group the largest IoU above threshold wins. No global optimisation,
        # so a candidate can claim a particle the optimum would have spent
        # elsewhere.
        major = inter.argmax(axis=1)
        val = gated[np.arange(gated.shape[0]), major]
        for j in np.unique(major):
            cand = np.nonzero((major == j) & (val > 0))[0]
            if len(cand):
                assigned[cand[np.argmax(val[cand])]] = True
    return assigned


def count_fakes(assigned, n_hits_cl, cut):
    """Unassigned and assigned candidate counts above a hit cut."""
    sel = n_hits_cl > cut
    n_match = int(assigned[sel].sum())
    return int(sel.sum()) - n_match, n_match


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache_path", required=True)
    ap.add_argument("--embed_dim", type=int, default=4)
    ap.add_argument("--tbeta", type=float, default=0.1)
    ap.add_argument("--td", type=float, default=0.2)
    ap.add_argument("--merge_td", type=float, default=0.0)
    ap.add_argument("--attach_td", type=float, default=0.0)
    ap.add_argument("--n_events", type=int, default=400)
    ap.add_argument("--cuts", default="0,3,10")
    args = ap.parse_args()

    cuts = [int(c) for c in args.cuts.split(",")]

    print(__doc__)
    print(f"cache={args.cache_path}  tbeta={args.tbeta} td={args.td} "
          f"merge={args.merge_td} attach={args.attach_td}\n")

    hits = pl.read_parquet(os.path.join(args.cache_path, "forward_hits.parquet"))
    cache = cache_from_dataframe(hits, args.embed_dim)[: args.n_events]
    del hits

    tot = {("hungarian", c): [0, 0] for c in cuts}
    tot.update({("greedy", c): [0, 0] for c in cuts})

    for k, e in enumerate(cache):
        labels = get_clustering_greedy(e["sig_beta"], e["sig_coords"],
                                       tbeta=args.tbeta, td=args.td)
        if args.merge_td > 0:
            labels = merge_fragments(labels, e["sig_coords"], e["sig_beta"],
                                     args.merge_td)
        if args.attach_td > 0:
            labels = attach_debris(labels, e["sig_coords"], args.attach_td)
        cl, pr, inter, n_hits_cl, _ = contingency(labels, e["sig_mc"])
        if not len(cl) or not len(pr):
            continue
        _, _, iou = iou_matrix(labels, e["sig_mc"])
        for mode in ("hungarian", "greedy"):
            a = assign(iou, inter, mode)
            for cut in cuts:
                f, m = count_fakes(a, n_hits_cl, cut)
                tot[(mode, cut)][0] += f
                tot[(mode, cut)][1] += m
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(cache)} events", flush=True)

    print(f"\n{'cut':>5} {'Hungarian':>22} {'greedy':>22} {'delta':>8}")
    print(f"{'':>5} {'fakes':>9}{'matched':>7}{'rate':>7} "
          f"{'fakes':>9}{'matched':>7}{'rate':>7}")
    print("-" * 60)
    for cut in cuts:
        hf, hm = tot[("hungarian", cut)]
        gf, gm = tot[("greedy", cut)]
        hr = 100 * hf / max(hm, 1)
        gr = 100 * gf / max(gm, 1)
        print(f"{'>' + str(cut):>5} {hf:9d}{hm:7d}{hr:6.1f}% "
              f"{gf:9d}{gm:7d}{gr:6.1f}% {gr - hr:+7.1f}")
    print()
    worst = max(abs(100 * tot[("greedy", c)][0] / max(tot[("greedy", c)][1], 1)
                    - 100 * tot[("hungarian", c)][0]
                    / max(tot[("hungarian", c)][1], 1)) for c in cuts)
    print(f"largest disagreement across hit cuts: {worst:.2f} percentage points")
    print("If this is small, the cached-aggregate emulation in "
          "ggtf_fake_rate.py can be quoted as their metric.")


if __name__ == "__main__":
    main()
