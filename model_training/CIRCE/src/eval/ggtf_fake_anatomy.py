"""Anatomy of the fake candidates and of the embedding, per checkpoint.

For each candidate at the given operating point, classify it (matched /
merged / fake per the GGTF rule) and, for fakes, identify what built it:
a fragment of a target already matched by another cluster (a clone), a
fragment of an unmatched target, a sub-3-hit stub particle, or noise hits.
Also reports embedding health: mean beta per hit class, and per-target
intra-cluster spread vs distance to the nearest other target's condensation
point (the fragmentation margin).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/home/marko.cechovic/cgatr-paper/literature/cgatr_fcc_pkg")
import numpy as np
import polars as pl
from multiprocessing import Pool

from src.eval_sweep_v33 import get_clustering_greedy

_G = {}


def _init(tbeta, td, mch):
    _G.update(tbeta=tbeta, td=td, mch=mch)


def _task(t):
    betas, coords, mc, gen1 = t
    labels = np.asarray(get_clustering_greedy(
        betas, coords, tbeta=_G["tbeta"], td=_G["td"]))
    if _G["mch"] > 1:
        keep, cnt = np.unique(labels[labels >= 0], return_counts=True)
        small = set(keep[cnt < _G["mch"]].tolist())
        if small:
            labels = np.array([-1 if v in small else v for v in labels], labels.dtype)
    real = labels >= 0
    csize = defaultdict(int)
    for lab in labels[real]:
        csize[int(lab)] += 1
    uniq, counts = np.unique(mc, return_counts=True)
    nhits = {int(p): int(c) for p, c in zip(uniq, counts)}
    targets = {p for p, c in nhits.items() if c >= 3}

    comp = defaultdict(lambda: defaultdict(int))     # cluster -> particle -> n
    for p, lab in zip(mc[real], labels[real]):
        comp[int(lab)][int(p)] += 1

    matched_clusters, matched_targets = set(), set()
    eff_over = defaultdict(int)
    for lab, d in comp.items():
        for p, n in d.items():
            if p in targets and n / csize[lab] > 0.75:
                matched_clusters.add(lab)
                matched_targets.add(p)
            if p in targets and n / nhits[p] > 0.75:
                eff_over[lab] += 1

    kinds = Counter()
    fake_sizes = []
    for lab, d in comp.items():
        if lab in matched_clusters:
            kinds["matched"] += 1
            continue
        if eff_over.get(lab, 0) > 1:
            kinds["merged"] += 1
            continue
        dom, ndom = max(d.items(), key=lambda kv: kv[1])
        fake_sizes.append(csize[lab])
        if dom in targets:
            if dom in matched_targets:
                kinds["fake_clone_of_matched"] += 1
            else:
                kinds["fake_fragment_unmatched"] += 1
        elif dom == 0:
            kinds["fake_noise_mc0"] += 1
        else:
            kinds["fake_stub"] += 1

    # beta by hit class; embedding margin per target
    beta_sum = Counter(); beta_n = Counter()
    for p, b in zip(mc, betas):
        p = int(p)
        cls = ("target_gen" if p in targets and p in gen1 else
               "target_sec" if p in targets else
               "noise_mc0" if p == 0 else "stub")
        beta_sum[cls] += float(b); beta_n[cls] += 1

    alphas, margins = {}, []
    for p in targets:
        m = mc == p
        i = np.argmax(np.where(m, betas, -1))
        alphas[p] = coords[i]
    ap = list(alphas.items())
    for p, a in ap:
        m = mc == p
        intra = float(np.sqrt(((coords[m] - a) ** 2).sum(1).mean()))
        others = [np.linalg.norm(a - b) for q, b in ap if q != p]
        if others:
            margins.append((min(others), intra))
    return kinds, fake_sizes, beta_sum, beta_n, margins


def main():
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument("--forward", type=Path, required=True)
    ap_.add_argument("--mc-signal", type=Path, required=True)
    ap_.add_argument("--seed-max", type=int, default=3)
    ap_.add_argument("--tbeta", type=float, default=0.6)
    ap_.add_argument("--td", type=float, default=0.3)
    ap_.add_argument("--min-cluster-hits", type=int, default=0)
    ap_.add_argument("--workers", type=int, default=16)
    ap_.add_argument("--out", type=Path)
    a = ap_.parse_args()

    mc_sig = pl.read_parquet(a.mc_signal,
                             columns=["mc_index", "event_id", "seed", "gen_status"])
    gen1 = defaultdict(set)
    for s, e, m in mc_sig.filter(pl.col("gen_status") == 1) \
                         .select(["seed", "event_id", "mc_index"]).iter_rows():
        gen1[(int(s), int(e))].add(int(m))
    cc = [f"coord_{i}" for i in range(4)]
    hits = pl.read_parquet(a.forward,
                           columns=["seed", "event_id", "beta", "mc_index"] + cc) \
             .filter(pl.col("seed") <= a.seed_max)
    tasks = []
    for (s, e), g in hits.group_by(["seed", "event_id"]):
        tasks.append((g["beta"].to_numpy(), g.select(cc).to_numpy(),
                      g["mc_index"].to_numpy(), gen1.get((int(s), int(e)), set())))
    print(f"{len(tasks)} events at {a.tbeta:g}/{a.td:g} mch={a.min_cluster_hits}",
          flush=True)

    K = Counter(); FS = []; BS = Counter(); BN = Counter(); MG = []
    with Pool(a.workers, initializer=_init,
              initargs=(a.tbeta, a.td, a.min_cluster_hits)) as pool:
        for kinds, fs, bs, bn, mg in pool.imap_unordered(_task, tasks, chunksize=8):
            K.update(kinds); FS.extend(fs); BS.update(bs); BN.update(bn); MG.extend(mg)

    tot = sum(K.values())
    fakes = sum(v for k, v in K.items() if k.startswith("fake"))
    print(f"\ncandidates {tot:,}; fake share {100*fakes/tot:.2f}%")
    for k in ("matched", "merged", "fake_clone_of_matched",
              "fake_fragment_unmatched", "fake_stub", "fake_noise_mc0"):
        print(f"  {k:>26}: {K[k]:>8,}  ({100*K[k]/tot:5.2f}% of candidates)")
    fs = np.array(FS)
    if len(fs):
        print(f"fake sizes: median {np.median(fs):.0f} hits, "
              f"p90 {np.percentile(fs, 90):.0f}, "
              f"share <4 hits {100*(fs < 4).mean():.1f}%, "
              f"share <=10 hits {100*(fs <= 10).mean():.1f}%")
    print("mean beta by hit class:",
          {k: round(BS[k]/BN[k], 3) for k in BN})
    mg = np.array(MG)
    if len(mg):
        ratio = mg[:, 0] / np.maximum(mg[:, 1], 1e-9)
        print(f"per-target margin (nearest other alpha / intra rms): "
              f"median {np.median(ratio):.2f}, "
              f"share < 1 {100*(ratio < 1).mean():.1f}%, "
              f"share < 2 {100*(ratio < 2).mean():.1f}%")
        print(f"intra rms median {np.median(mg[:,1]):.3f}; "
              f"nearest-alpha median {np.median(mg[:,0]):.3f} "
              f"(td for reference: {a.td:g})")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({
            "kinds": dict(K), "n_candidates": tot,
            "beta_by_class": {k: BS[k]/BN[k] for k in BN},
            "margin_median": float(np.median(ratio)) if len(mg) else None,
            "tbeta": a.tbeta, "td": a.td, "mch": a.min_cluster_hits,
            "seed_max": a.seed_max}, indent=2) + "\n")


if __name__ == "__main__":
    main()
