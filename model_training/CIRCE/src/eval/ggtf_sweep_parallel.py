"""Parallel clustering sweep under the GGTF-exact metric, for large caches.

Same protocol as ggtf_clustering_sweep.py (fake rate never reported without
efficiency; selection on the odd event half, the winner re-scored on the even
half), but parallelised one-config-per-worker so a 5,000-event subset finishes
in tens of minutes instead of a day. Events are loaded once in the parent and
shared with workers via fork. HDBSCAN is part of THIS grid, passed correctly:
cluster_event maps min_hits -> min_cluster_size/min_samples and td ->
cluster_selection_epsilon (0 here), the mapping that bit us on 2026-08-31.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, "/home/marko.cechovic/cgatr-paper/literature/cgatr_fcc_pkg")
import polars as pl

from src.eval.ggtf_clustering_sweep import load_events, score

G = {}


def build_configs():
    cfgs = []
    for clusterer in ("self_seed_greedy", "greedy"):
        for tb in (0.1, 0.3, 0.6):
            for td in (0.1, 0.15, 0.2, 0.3, 0.5, 0.8):
                for mch in (0, 4):
                    cfgs.append((clusterer, tb, td, mch))
    for eps in (0.10, 0.15, 0.20, 0.30, 0.50):
        for mch in (0, 4):
            cfgs.append(("dbscan", 0.0, eps, mch))
    for mcs in (5, 10, 20):
        cfgs.append(("hdbscan", 0.0, 0.0, mcs))  # min_hits -> min_cluster_size
    return cfgs


def _score_select(cfg):
    clusterer, tb, td, mch = cfg
    s = score(G["select"], clusterer, tb, td, mch, 3, G["keep"])
    return {"clusterer": clusterer, "tbeta": tb, "td": td,
            "min_cluster_hits": mch, **s}


def _score_on(events_key, cfg):
    clusterer, tb, td, mch = cfg
    return score(G[events_key], clusterer, tb, td, mch, 3, G["keep"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forward", type=Path, required=True)
    ap.add_argument("--mc-signal", type=Path, required=True)
    ap.add_argument("--seed-max", type=int, default=10,
                    help="use seeds 1..N of the cache")
    ap.add_argument("--generator-only", action="store_true")
    ap.add_argument("--min-efficiency", type=float, default=0.90)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    keep = None
    if args.generator_only:
        mc = pl.read_parquet(args.mc_signal,
                             columns=["mc_index", "event_id", "seed", "gen_status"])
        g1 = mc.filter(pl.col("gen_status") == 1)
        keep = set(zip(g1["seed"].to_list(), g1["event_id"].to_list(),
                       g1["mc_index"].to_list()))

    # filter the big cache to the subset once, in-memory
    full = pl.read_parquet(args.forward)
    sub_path = Path(str(args.out) + ".events.parquet")
    full.filter(pl.col("seed") <= args.seed_max).write_parquet(sub_path)
    del full
    events = load_events(sub_path, 4)
    sub_path.unlink()
    G["select"] = [e for i, e in enumerate(events) if i % 2 == 1]
    G["confirm"] = [e for i, e in enumerate(events) if i % 2 == 0]
    G["all"] = events
    G["keep"] = keep
    print(f"{len(events)} events (seeds <= {args.seed_max}) -> "
          f"{len(G['select'])} select / {len(G['confirm'])} confirm; "
          f"generator_only={args.generator_only}", flush=True)

    cfgs = build_configs()
    rows = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_score_select, cfgs, chunksize=1)):
            rows.append(r)
            print(f"  [{i+1}/{len(cfgs)}] {r['clusterer']:<16} tb={r['tbeta']:.2f} "
                  f"td/eps={r['td']:.2f} mch={r['min_cluster_hits']} -> "
                  f"fake {100*r['fake_rate']:6.2f}%  eff {100*r['efficiency']:6.2f}%",
                  flush=True)

    frontier = [r for r in rows if not any(
        o["efficiency"] > r["efficiency"] and o["fake_rate"] < r["fake_rate"]
        for o in rows)]
    frontier.sort(key=lambda r: r["efficiency"])
    print("\nPareto frontier (selection half):", flush=True)
    for r in frontier:
        print(f"  {r['clusterer']:<16} tb={r['tbeta']:.2f} td/eps={r['td']:.2f} "
              f"mch={r['min_cluster_hits']}  fake {100*r['fake_rate']:6.2f}%  "
              f"eff {100*r['efficiency']:6.2f}%  cand/ev {r['candidates_per_event']:.1f}",
              flush=True)

    best_eff = max(r["efficiency"] for r in rows)
    floor = args.min_efficiency
    if best_eff < floor:
        floor = best_eff - 0.01
        print(f"\nefficiency floor relaxed to {100*floor:.2f}% "
              f"(best reachable {100*best_eff:.2f}%)", flush=True)
    best = min((r for r in rows if r["efficiency"] >= floor),
               key=lambda r: r["fake_rate"])
    bcfg = (best["clusterer"], best["tbeta"], best["td"], best["min_cluster_hits"])
    print(f"\nwinner on selection half: {bcfg}  "
          f"fake {100*best['fake_rate']:.2f}% eff {100*best['efficiency']:.2f}%",
          flush=True)
    conf = _score_on("confirm", bcfg)
    alln = _score_on("all", bcfg)
    base = _score_on("all", ("greedy", 0.6, 0.3, 0))
    print(f"winner on held-out half : fake {100*conf['fake_rate']:.2f}%  "
          f"eff {100*conf['efficiency']:.2f}%", flush=True)
    print(f"winner on all events    : fake {100*alln['fake_rate']:.2f}%  "
          f"eff {100*alln['efficiency']:.2f}%", flush=True)
    print(f"0.6/0.3 ref, all events : fake {100*base['fake_rate']:.2f}%  "
          f"eff {100*base['efficiency']:.2f}%", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"grid": rows, "frontier": frontier, "best_on_selection": best,
         "confirm_half": conf, "all_events": alln, "baseline_0.6_0.3": base,
         "generator_only": args.generator_only, "seed_max": args.seed_max,
         "n_events": len(events)}, indent=2, default=float) + "\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
