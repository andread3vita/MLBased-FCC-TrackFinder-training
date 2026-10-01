"""Reproduce GGTF's own fake-rate calculation, line for line, on our cache.

Transcribed from their code, not from our reading of it:

  * clustering            `object_cond.py:817  get_clustering(betas, X, tbeta, td)`
                          beta-greedy NMS, identical in algorithm to ours.
                          Their inference calls it at tbeta=0.6, td=0.3
                          (`inference_oc_tracks.py:82`).
  * per (particle, cluster)
        efficiency[p][l] = hits of p inside l / all hits of p
        purity[p][l]     = hits of p inside l / all hits of l
                          (`inference_oc_tracks.py`, match_tracks)
  * matching             a particle is matched to cluster l iff
                          **purity[p][l] > 0.75**. No efficiency requirement:
                          the `eff > 0.5 and pur > 0.5` line is commented out in
                          their source. Several clusters may match one particle,
                          so clones are all recorded against it.
  * fake candidates      clusters matched to *no* particle.
  * fake vs merged       for an unmatched cluster, count particles with
                          efficiency > 0.75 into it. More than one and the
                          cluster is a **merge** and is not counted as fake;
                          otherwise it is a fake
                          (notebook `1_evaluation_IDEA_tracking.ipynb`, cell 3).
  * rate                 num_fake_tracks / num_reco_tracks, where
                          num_reco_tracks sums the largest cluster label per
                          event, i.e. the number of reconstructed candidates.
                          It is a fraction of candidates and is bounded by 1.

Two things this establishes that our own `ggtf_fake_rate_gt*` got wrong. Their
fake rate uses **no Hungarian assignment** - `linear_sum_assignment` is imported
in their notebook and never called - and its denominator is *all* candidates
rather than the matched ones, so it is not unbounded and does not read 90% on a
fragmenting model. It also applies **no theta, pT or hit cut**: the cuts in
`trackingEfficiencyPlot` bound the efficiency denominator only.

Their target set is whatever survives `create_garbage_label(minNumHits=3)`,
which relabels particles under three hits, and hits flagged secondary, to noise
while leaving the hits in the graph. `--min-target-hits` reproduces the first;
`--generator-only` reproduces the second through gen_status, which is where our
production records it (FINDINGS M83).

  PYTHONPATH=. python src/eval/ggtf_exact_fake_rate.py \
      --forward <emb/forward_hits.parquet> --mc-signal <mc_signal.parquet> \
      --tbeta 0.1 --td 0.2
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from src.eval_sweep_v33 import get_clustering_greedy

NOISE = -1


def cluster_event(betas, coords, tbeta, td, min_cluster_hits):
    """Their get_clustering, plus our optional minimum candidate size.

    Their pipeline has no minimum, so min_cluster_hits=0 is the faithful
    setting; ours drops candidates below four hits at the reference point.
    """
    labels = np.asarray(get_clustering_greedy(betas, coords, tbeta=tbeta, td=td))
    if min_cluster_hits > 1:
        keep, counts = np.unique(labels[labels >= 0], return_counts=True)
        small = set(keep[counts < min_cluster_hits].tolist())
        if small:
            labels = np.array([NOISE if v in small else v for v in labels],
                              dtype=labels.dtype)
    return labels


def event_counts(labels, mc, target_mask):
    """Score one event under Andrea's exact matching rule (1_evaluation_IDEA_tracking.ipynb).

    Returns (n_candidates, n_fake, n_merged, n_matched_clusters, n_targets,
    n_matched_targets). The last two are their efficiency numerator and
    denominator: `trackingEfficiencyPlot` counts a particle as reconstructed
    when its `trackLabel` list is non-empty, and that list is exactly the set of
    clusters matching it at purity > 0.75.

    `target_mask` marks hits whose particle is a target. Non-target hits stay in
    the event and still count in cluster sizes, exactly as their relabel-to-noise
    keeps them in the graph. In Andrea's exact formula, cluster purity is evaluated
    against ALL true particles in the event (p >= 0), so a cluster is only fake if
    no true particle contributes > 75% of its hits and it is not merged.
    """
    real = labels >= 0
    cluster_sizes = defaultdict(int)
    for lab in labels[real]:
        cluster_sizes[int(lab)] += 1

    uniq, counts = np.unique(mc, return_counts=True)
    nhits_all = {int(p): int(c) for p, c in zip(uniq, counts)}

    particle_hits = defaultdict(int)
    for p in mc[target_mask]:
        particle_hits[int(p)] += 1
    n_targets = len(particle_hits)
    if not cluster_sizes:
        return 0, 0, 0, 0, n_targets, 0

    # Overlap with target particles for efficiency numerator
    overlap_target = defaultdict(int)
    both = real & target_mask
    for p, lab in zip(mc[both], labels[both]):
        overlap_target[(int(p), int(lab))] += 1

    matched_targets = set()
    for (p, lab), n in overlap_target.items():
        if n / cluster_sizes[lab] > 0.75:
            matched_targets.add(p)

    # Overlap with ALL true particles (mc >= 0) for candidate purity / fake rate (Andrea's formula)
    overlap_all = defaultdict(int)
    for p, lab in zip(mc[real], labels[real]):
        if p >= 0:
            overlap_all[(int(p), int(lab))] += 1

    matched_clusters = set()
    eff_over = defaultdict(int)                     # cluster -> particles at eff>0.75
    for (p, lab), n in overlap_all.items():
        if n / cluster_sizes[lab] > 0.75:           # their matching rule
            matched_clusters.add(lab)
        if n / nhits_all[p] > 0.75:
            eff_over[lab] += 1

    n_fake = n_merged = 0
    for lab in cluster_sizes:
        if lab in matched_clusters:
            continue
        if eff_over.get(lab, 0) > 1:                # covers several particles
            n_merged += 1
        else:
            n_fake += 1
    return (len(cluster_sizes), n_fake, n_merged, len(matched_clusters),
            n_targets, len(matched_targets))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forward", type=Path, required=True)
    ap.add_argument("--mc-signal", type=Path, required=True)
    ap.add_argument("--embed-dim", type=int, default=4)
    ap.add_argument("--tbeta", type=float, default=0.1)
    ap.add_argument("--td", type=float, default=0.2)
    ap.add_argument("--min-cluster-hits", type=int, default=0,
                    help="0 reproduces GGTF, who apply no candidate size cut")
    ap.add_argument("--min-target-hits", type=int, default=3,
                    help="their create_garbage_label(minNumHits=3)")
    ap.add_argument("--generator-only", action="store_true",
                    help="drop gen_status != 1 from the target set, which is "
                         "how our production records their secondary flag (M83)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    hits = pl.read_parquet(args.forward)
    coord_cols = [f"coord_{i}" for i in range(args.embed_dim)]

    keep_particles = None
    if args.generator_only:
        mc = pl.read_parquet(args.mc_signal,
                             columns=["mc_index", "event_id", "seed", "gen_status"])
        gen1 = mc.filter(pl.col("gen_status") == 1)
        keep_particles = set(
            zip(gen1["seed"].to_list(), gen1["event_id"].to_list(),
                gen1["mc_index"].to_list())
        )

    tot_cand = tot_fake = tot_merged = tot_matched = 0
    tot_targets = tot_matched_targets = 0
    n_events = 0
    for (seed, event_id), grp in hits.group_by(["seed", "event_id"]):
        betas = grp["beta"].to_numpy()
        coords = grp.select(coord_cols).to_numpy()
        mc = grp["mc_index"].to_numpy()

        labels = cluster_event(betas, coords, args.tbeta, args.td,
                               args.min_cluster_hits)

        # target set: their minNumHits rule, then optionally generator-only
        uniq, counts = np.unique(mc, return_counts=True)
        big_enough = {int(p) for p, c in zip(uniq, counts)
                      if c >= args.min_target_hits}
        if keep_particles is not None:
            big_enough = {p for p in big_enough
                          if (int(seed), int(event_id), p) in keep_particles}
        target_mask = np.array([int(p) in big_enough for p in mc])

        cand, fake, merged, matched, n_tgt, n_mtgt = event_counts(
            labels, mc, target_mask)
        tot_cand += cand
        tot_fake += fake
        tot_merged += merged
        tot_matched += matched
        tot_targets += n_tgt
        tot_matched_targets += n_mtgt
        n_events += 1

    rate = tot_fake / tot_cand if tot_cand else float("nan")
    merge_rate = tot_merged / tot_cand if tot_cand else float("nan")
    efficiency = tot_matched_targets / tot_targets if tot_targets else float("nan")
    result = {
        "definition": "GGTF notebook cell 3: num_fake_tracks / num_reco_tracks",
        "matching": "particle matched to cluster iff purity > 0.75",
        "fake": "cluster matched to no particle and covering <=1 particle at efficiency > 0.75",
        "merged": "cluster matched to no particle but covering >1 particle at efficiency > 0.75",
        "tbeta": args.tbeta, "td": args.td,
        "min_cluster_hits": args.min_cluster_hits,
        "min_target_hits": args.min_target_hits,
        "generator_only": args.generator_only,
        "n_events": n_events,
        "n_candidates": tot_cand,
        "n_matched_clusters": tot_matched,
        "n_fake": tot_fake,
        "n_merged": tot_merged,
        "ggtf_fake_rate": rate,
        "ggtf_merge_rate": merge_rate,
        "n_targets": tot_targets,
        "n_matched_targets": tot_matched_targets,
        "ggtf_efficiency": efficiency,
        "candidates_per_event": tot_cand / max(n_events, 1),
    }
    print(json.dumps(result, indent=2))
    print(f"\nFake Rate : {tot_fake} / {tot_cand} = {rate:.4f}")
    print(f"Merge Rate: {tot_merged} / {tot_cand} = {merge_rate:.4f}")
    print(f"Efficiency: {tot_matched_targets} / {tot_targets} = {efficiency:.4f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
