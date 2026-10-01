"""Best fake rate reachable under GGTF's own definition, by tuning clustering.

Scores every clustering configuration with the transcription in
`ggtf_exact_fake_rate.py`, so the metric is theirs throughout and only the
reconstruction moves.

Two guards, both learned the hard way:

  * **The fake rate is reported against efficiency, never alone.** Their rate is
    fakes over candidates, so any clustering that emits almost nothing scores
    almost zero. A conservative point can read 2% while finding half the tracks.
    The selection therefore maximises efficiency subject to a fake-rate ceiling,
    or minimises fake rate subject to an efficiency floor, and both columns are
    always printed.
  * **Selection and confirmation are split.** M85 found that two of three
    operating points chosen on 500 events reversed on an independent sample, one
    of them from 2.4% to 10.2% fake rate. Events are split by parity here:
    configurations are ranked on the odd half and the winner is re-scored on the
    even half, which is the number to quote.

  PYTHONPATH=. python src/eval/ggtf_clustering_sweep.py \
      --forward <emb/forward_hits.parquet> --mc-signal <mc_signal.parquet>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.eval.fcc_cache_parallel import cluster_event as apply_clusterer
from src.eval.ggtf_exact_fake_rate import event_counts

NOISE = -1


def load_events(forward: Path, embed_dim: int):
    hits = pl.read_parquet(forward)
    coord_cols = [f"coord_{i}" for i in range(embed_dim)]
    events = []
    for (seed, event_id), grp in hits.group_by(["seed", "event_id"]):
        events.append({
            "seed": int(seed), "event_id": int(event_id),
            "betas": grp["beta"].to_numpy(),
            "coords": grp.select(coord_cols).to_numpy(),
            "mc": grp["mc_index"].to_numpy(),
        })
    events.sort(key=lambda e: (e["seed"], e["event_id"]))
    return events


def target_mask_for(mc, min_target_hits, keep_particles, seed, event_id):
    uniq, counts = np.unique(mc, return_counts=True)
    keep = {int(p) for p, c in zip(uniq, counts) if c >= min_target_hits}
    if keep_particles is not None:
        keep = {p for p in keep if (seed, event_id, p) in keep_particles}
    return np.array([int(p) in keep for p in mc])


def score(events, clusterer, tbeta, td, min_cluster_hits,
          min_target_hits, keep_particles):
    tot = dict(cand=0, fake=0, merged=0, targets=0, matched_targets=0)
    for ev in events:
        labels = np.asarray(apply_clusterer(
            clusterer, ev["betas"], ev["coords"], tbeta, td, min_cluster_hits))
        if min_cluster_hits > 1:
            keep, counts = np.unique(labels[labels >= 0], return_counts=True)
            small = set(keep[counts < min_cluster_hits].tolist())
            if small:
                labels = np.array([NOISE if v in small else v for v in labels],
                                  dtype=labels.dtype)
        tmask = target_mask_for(ev["mc"], min_target_hits, keep_particles,
                                ev["seed"], ev["event_id"])
        cand, fake, merged, _m, n_tgt, n_mtgt = event_counts(
            labels, ev["mc"], tmask)
        tot["cand"] += cand
        tot["fake"] += fake
        tot["merged"] += merged
        tot["targets"] += n_tgt
        tot["matched_targets"] += n_mtgt
    fake_rate = tot["fake"] / tot["cand"] if tot["cand"] else float("nan")
    eff = tot["matched_targets"] / tot["targets"] if tot["targets"] else float("nan")
    return {**tot, "fake_rate": fake_rate, "efficiency": eff,
            "candidates_per_event": tot["cand"] / max(len(events), 1)}


def build_grid():
    grid = []
    for clusterer in ("self_seed_greedy", "greedy"):
        for tbeta in (0.1, 0.3, 0.6):
            for td in (0.1, 0.15, 0.2, 0.3, 0.5, 0.8):
                grid.append((clusterer, tbeta, td))
    for eps in (0.10, 0.15, 0.20, 0.30, 0.50):
        grid.append(("dbscan", 0.0, eps))
    # HDBSCAN is deliberately NOT swept through this grid: cluster_event maps
    # the third argument to cluster_selection_epsilon, not min_cluster_size,
    # so a "min_cluster_size" passed here silently becomes an epsilon of 5-20
    # in an embedding of scale ~1 and fuses everything into ~2 clusters per
    # event. That mistake was made once (2026-08-31) and produced fake-rate
    # rows of 0% at 8% efficiency. Score HDBSCAN by calling it directly with
    # min_cluster_size/min_samples instead.
    return grid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forward", type=Path, required=True)
    ap.add_argument("--mc-signal", type=Path, required=True)
    ap.add_argument("--embed-dim", type=int, default=4)
    ap.add_argument("--min-target-hits", type=int, default=3)
    ap.add_argument("--generator-only", action="store_true")
    ap.add_argument("--min-efficiency", type=float, default=0.90,
                    help="floor for the 'best fake rate' pick")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    keep_particles = None
    if args.generator_only:
        mc = pl.read_parquet(args.mc_signal,
                             columns=["mc_index", "event_id", "seed", "gen_status"])
        g1 = mc.filter(pl.col("gen_status") == 1)
        keep_particles = set(zip(g1["seed"].to_list(), g1["event_id"].to_list(),
                                 g1["mc_index"].to_list()))

    events = load_events(args.forward, args.embed_dim)
    select = [e for i, e in enumerate(events) if i % 2 == 1]   # odd: choose here
    confirm = [e for i, e in enumerate(events) if i % 2 == 0]  # even: quote here
    print(f"{len(events)} events -> {len(select)} select / {len(confirm)} confirm\n")

    rows = []
    for clusterer, tbeta, td in build_grid():
        for mch in (0, 4):
            s = score(select, clusterer, tbeta, td, mch,
                      args.min_target_hits, keep_particles)
            rows.append({"clusterer": clusterer, "tbeta": tbeta, "td": td,
                         "min_cluster_hits": mch, **s})

    # Pareto frontier: nothing else is both more efficient and less fake. Sorting
    # by fake rate alone would crown a clusterer that emits almost nothing.
    frontier = [r for r in rows if not any(
        o["efficiency"] > r["efficiency"] and o["fake_rate"] < r["fake_rate"]
        for o in rows)]
    frontier.sort(key=lambda r: r["efficiency"])
    print(f"{'clusterer':<18}{'tbeta':>6}{'td/eps':>8}{'mch':>5}"
          f"{'fake':>8}{'eff':>8}{'cand/ev':>9}   (frontier, selection half)")
    for r in frontier:
        print(f"{r['clusterer']:<18}{r['tbeta']:>6.2f}{r['td']:>8.2f}"
              f"{r['min_cluster_hits']:>5}{100*r['fake_rate']:>8.2f}"
              f"{100*r['efficiency']:>8.2f}{r['candidates_per_event']:>9.1f}")

    best_eff = max(r["efficiency"] for r in rows)
    floor = args.min_efficiency
    if best_eff < floor:
        # Never fall back to "lowest fake rate overall" -- that is the degenerate
        # corner. Hold efficiency within a point of the best reachable instead.
        floor = best_eff - 0.01
        print(f"\nNo configuration reaches efficiency {args.min_efficiency:.0%}; "
              f"best reachable is {100*best_eff:.2f}%, so the floor is relaxed to "
              f"{100*floor:.2f}% rather than dropped.")
    eligible = [r for r in rows if r["efficiency"] >= floor]
    best = min(eligible, key=lambda r: r["fake_rate"])
    print(f"\nbest fake rate at efficiency >= {floor:.1%} "
          f"(chosen on the selection half):")
    print(f"  {best['clusterer']} tbeta={best['tbeta']} td={best['td']} "
          f"min_cluster_hits={best['min_cluster_hits']}")
    print(f"  fake {100*best['fake_rate']:.2f}%  eff {100*best['efficiency']:.2f}%")

    conf = score(confirm, best["clusterer"], best["tbeta"], best["td"],
                 best["min_cluster_hits"], args.min_target_hits, keep_particles)
    print(f"\nsame point re-scored on the held-out half  <-- quote this")
    print(f"  fake {100*conf['fake_rate']:.2f}%  eff {100*conf['efficiency']:.2f}%  "
          f"cand/ev {conf['candidates_per_event']:.1f}")
    full = score(events, best["clusterer"], best["tbeta"], best["td"],
                 best["min_cluster_hits"], args.min_target_hits, keep_particles)
    print(f"all 200 events at that point: fake {100*full['fake_rate']:.2f}%  "
          f"eff {100*full['efficiency']:.2f}%")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"grid": rows, "best_on_selection": best,
             "confirm_half": conf, "all_events": full,
             "min_efficiency": args.min_efficiency,
             "generator_only": args.generator_only,
             "min_target_hits": args.min_target_hits}, indent=2) + "\n")


if __name__ == "__main__":
    main()
