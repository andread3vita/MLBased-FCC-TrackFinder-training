"""Single-pass GGTF-exact evaluation for large caches.

The per-metric CLIs (ggtf_exact_fake_rate, ggtf_efficiency_grid, the eff-vs-pT
plot) each re-cluster every event, which is fine at hundreds of events and
prohibitive at 50,000. This clusters each event exactly once, in parallel, and
derives in the same pass:

  * fake / merge / matched-target counts with secondaries-in targets
    (all particles with >= 3 hits),
  * the same with generator-only targets (gen_status == 1),
  * per-target rows (nhits, gen_status, pt, theta, matched) for the
    efficiency grid and the eff-vs-pT plot.

Matching logic is identical to ggtf_exact_fake_rate.event_counts: a particle
is matched to a cluster iff purity > 0.75; unmatched clusters covering more
than one target at hit-efficiency > 0.75 are merged, otherwise fake. The
matched flag of a target does not depend on the target convention, so one
overlap computation serves all outputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/home/marko.cechovic/cgatr-paper/literature/cgatr_fcc_pkg")
import numpy as np
import polars as pl
from multiprocessing import Pool

from src.eval_sweep_v33 import get_clustering_greedy

_G = {}


def _init(tbeta, td, mch):
    _G["tbeta"], _G["td"], _G["mch"] = tbeta, td, mch


def _convention_counts(cluster_sizes, overlap_all, overlap_targets, nhits_all, targets):
    matched_targets = set()
    for (p, lab), n in overlap_targets.items():
        if n / cluster_sizes[lab] > 0.75:
            matched_targets.add(p)

    matched_clusters = set()
    eff_over = defaultdict(int)
    for (p, lab), n in overlap_all.items():
        if n / cluster_sizes[lab] > 0.75:
            matched_clusters.add(lab)
        if n / nhits_all[p] > 0.75:
            eff_over[lab] += 1
    n_fake = n_merged = 0
    for lab in cluster_sizes:
        if lab in matched_clusters:
            continue
        if eff_over.get(lab, 0) > 1:
            n_merged += 1
        else:
            n_fake += 1
    return (len(cluster_sizes), n_fake, n_merged, len(matched_clusters),
            len(targets), len(matched_targets))


def _event_task(task):
    seed, event_id, betas, coords, mc, gen1 = task
    labels = np.asarray(get_clustering_greedy(
        betas, coords, tbeta=_G["tbeta"], td=_G["td"]))
    if _G["mch"] > 1:
        keep, cnt = np.unique(labels[labels >= 0], return_counts=True)
        small = set(keep[cnt < _G["mch"]].tolist())
        if small:
            labels = np.array([-1 if v in small else v for v in labels],
                              dtype=labels.dtype)
    real = labels >= 0
    cluster_sizes = defaultdict(int)
    for lab in labels[real]:
        cluster_sizes[int(lab)] += 1

    uniq, counts = np.unique(mc, return_counts=True)
    nhits = {int(p): int(c) for p, c in zip(uniq, counts)}
    targets_all = {p for p, c in nhits.items() if c >= 3}
    targets_gen = {p for p in targets_all if p in gen1}

    overlap_all = defaultdict(int)
    overlap_all_targets = defaultdict(int)
    overlap_gen_targets = defaultdict(int)
    if cluster_sizes:
        for p, lab in zip(mc[real], labels[real]):
            if p >= 0:
                p_int = int(p)
                overlap_all[(p_int, int(lab))] += 1
                if p_int in targets_all:
                    overlap_all_targets[(p_int, int(lab))] += 1
                if p_int in targets_gen:
                    overlap_gen_targets[(p_int, int(lab))] += 1

    c_all = _convention_counts(cluster_sizes, overlap_all, overlap_all_targets, nhits, targets_all) \
        if cluster_sizes else (0, 0, 0, 0, len(targets_all), 0)
    c_gen = _convention_counts(cluster_sizes, overlap_all, overlap_gen_targets, nhits, targets_gen) \
        if cluster_sizes else (0, 0, 0, 0, len(targets_gen), 0)

    matched_p = {p for (p, lab), n in overlap_all_targets.items()
                 if n / cluster_sizes[lab] > 0.75}
    rows = [(seed, event_id, p, nhits[p], p in matched_p) for p in targets_all]
    return c_all, c_gen, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forward", type=Path, required=True)
    ap.add_argument("--mc-signal", type=Path, required=True)
    ap.add_argument("--embed-dim", type=int, default=4)
    ap.add_argument("--tbeta", type=float, default=0.6)
    ap.add_argument("--td", type=float, default=0.3)
    ap.add_argument("--min-cluster-hits", type=int, default=0,
                    help="relabel candidates smaller than this to noise "
                         "(0 = GGTF-faithful, no cut)")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    mc_sig = pl.read_parquet(args.mc_signal,
                             columns=["mc_index", "event_id", "seed",
                                      "gen_status", "pt", "theta"])
    gen1 = defaultdict(set)
    for s, e, m in mc_sig.filter(pl.col("gen_status") == 1) \
                         .select(["seed", "event_id", "mc_index"]).iter_rows():
        gen1[(int(s), int(e))].add(int(m))

    coord_cols = [f"coord_{i}" for i in range(args.embed_dim)]
    hits = pl.read_parquet(args.forward,
                           columns=["seed", "event_id", "beta", "mc_index"] + coord_cols)
    tasks = []
    for (seed, event_id), grp in hits.group_by(["seed", "event_id"]):
        key = (int(seed), int(event_id))
        tasks.append((key[0], key[1], grp["beta"].to_numpy(),
                      grp.select(coord_cols).to_numpy(),
                      grp["mc_index"].to_numpy(), gen1.get(key, set())))
    del hits
    print(f"{len(tasks)} events, {args.workers} workers, "
          f"tbeta/td {args.tbeta:g}/{args.td:g}", flush=True)

    tot_all = np.zeros(6, dtype=np.int64)
    tot_gen = np.zeros(6, dtype=np.int64)
    all_rows = []
    with Pool(args.workers, initializer=_init, initargs=(args.tbeta, args.td, args.min_cluster_hits)) as pool:
        for i, (c_all, c_gen, rows) in enumerate(
                pool.imap_unordered(_event_task, tasks, chunksize=16)):
            tot_all += np.array(c_all)
            tot_gen += np.array(c_gen)
            all_rows.extend(rows)
            if (i + 1) % 5000 == 0:
                print(f"  ... {i + 1} events", flush=True)

    def report(tot, name):
        cand, fake, merged, mcl, tgt, mtgt = (int(x) for x in tot)
        res = {"tbeta": args.tbeta, "td": args.td, "min_cluster_hits": args.min_cluster_hits, "n_events": len(tasks),
               "n_candidates": cand, "n_fake": fake, "n_merged": merged,
               "n_matched_clusters": mcl, "n_targets": tgt,
               "n_matched_targets": mtgt,
               "ggtf_fake_rate": fake / cand if cand else None,
               "ggtf_merge_rate": merged / cand if cand else None,
               "ggtf_efficiency": mtgt / tgt if tgt else None,
               "candidates_per_event": cand / len(tasks)}
        (args.outdir / f"fake_rate_{args.tbeta:g}_{args.td:g}_{name}.json").write_text(
            json.dumps(res, indent=2) + "\n")

        def pct(key):  # tgt/cand can be 0 (e.g. no generator-only targets on a tiny slice)
            v = res[key]
            return f"{100*v:.2f}%" if v is not None else "n/a"

        print(f"[{name}] fake {pct('ggtf_fake_rate')}  "
              f"merge {pct('ggtf_merge_rate')}  "
              f"matched {pct('ggtf_efficiency')} "
              f"({mtgt:,}/{tgt:,})  cand/ev {res['candidates_per_event']:.2f}",
              flush=True)

    report(tot_all, "secondaries_in")
    report(tot_gen, "generator_only")

    rows_df = pl.DataFrame(all_rows, orient="row",
                           schema=["seed", "event_id", "mc_index", "nhits", "matched"])
    rows_df = rows_df.join(mc_sig, on=["seed", "event_id", "mc_index"], how="left") \
                     .with_columns((pl.col("theta").degrees()).alias("theta_deg"))
    rows_df.write_parquet(args.outdir / "target_rows.parquet")

    cut = rows_df.filter((pl.col("pt") > 0) & (pl.col("theta_deg") > 15)
                         & (pl.col("theta_deg") < 165))
    grid = {}
    print(f"efficiency grid, 15 < theta < 165, pT > 0:", flush=True)
    for gens in ([1], [0, 1]):
        for nmin in (3, 10):
            sub = cut.filter(pl.col("gen_status").is_in(gens)
                             & (pl.col("nhits") > nmin))
            eff = sub["matched"].mean()  # None on an empty slice (polars empty-mean)
            key = f"genStatus={gens},nHits>{nmin}"
            grid[key] = {"efficiency": eff, "n_targets": sub.height}
            eff_str = f"{100*eff:6.2f}%" if eff is not None else "   n/a"
            print(f"  {key:>26}: {eff_str}  ({sub.height:,} targets)", flush=True)
    (args.outdir / f"efficiency_grid_{args.tbeta:g}_{args.td:g}.json").write_text(
        json.dumps({"tbeta": args.tbeta, "td": args.td,
                    "n_events": len(tasks), "grid": grid}, indent=2) + "\n")


if __name__ == "__main__":
    main()
