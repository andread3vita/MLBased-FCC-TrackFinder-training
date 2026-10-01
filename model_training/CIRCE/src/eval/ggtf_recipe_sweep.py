"""Does fragment merging fix the GGTF-convention fake rate? (M92 follow-up)

The fake anatomy showed 77% of ep24's fakes are fragments of real tracks.
This sweeps greedy base operating points x merge_fragments radii (the star
merge from the champion recipe) with an optional attach stage and candidate
size cut, scored under the GGTF-exact metric with the split-half protocol.
Design slice only (seeds 1-10); seeds 11-100 stay untouched for final numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, "/home/marko.cechovic/cgatr-paper/literature/cgatr_fcc_pkg")
import numpy as np
import polars as pl

from src.eval_sweep_v33 import get_clustering_greedy
from src.eval.fcc_cache_parallel import merge_fragments, attach_debris
from src.eval.ggtf_exact_fake_rate import event_counts

G = {}


def _score(cfg):
    tb, td, tmerge, tattach, mch = cfg
    tot = dict(cand=0, fake=0, merged=0, targets=0, matched=0)
    for ev in G["events"]:
        labels = np.asarray(get_clustering_greedy(
            ev["betas"], ev["coords"], tbeta=tb, td=td))
        if tmerge > 0:
            labels = np.asarray(merge_fragments(labels, ev["coords"], ev["betas"], tmerge))
        if tattach > 0:
            labels = np.asarray(attach_debris(labels, ev["coords"], tattach))
        if mch > 1:
            keep, cnt = np.unique(labels[labels >= 0], return_counts=True)
            small = set(keep[cnt < mch].tolist())
            if small:
                labels = np.array([-1 if v in small else v for v in labels], labels.dtype)
        uniq, counts = np.unique(ev["mc"], return_counts=True)
        big = {int(p) for p, c in zip(uniq, counts) if c >= 3}
        tmask = np.isin(ev["mc"], np.fromiter(big, int, len(big)))
        cand, fake, merged, _m, n_t, n_mt = event_counts(labels, ev["mc"], tmask)
        tot["cand"] += cand; tot["fake"] += fake; tot["merged"] += merged
        tot["targets"] += n_t; tot["matched"] += n_mt
    return {"tbeta": tb, "td": td, "merge_td": tmerge, "attach_td": tattach,
            "mch": mch,
            "fake_rate": tot["fake"]/tot["cand"] if tot["cand"] else float("nan"),
            "merge_rate": tot["merged"]/tot["cand"] if tot["cand"] else float("nan"),
            "efficiency": tot["matched"]/tot["targets"],
            "cand_per_ev": tot["cand"]/len(G["events"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forward", type=Path, required=True)
    ap.add_argument("--seed-max", type=int, default=10)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    cc = [f"coord_{i}" for i in range(4)]
    hits = pl.read_parquet(a.forward,
                           columns=["seed", "event_id", "beta", "mc_index"] + cc) \
             .filter(pl.col("seed") <= a.seed_max)
    events = []
    for (s, e), g in hits.group_by(["seed", "event_id"]):
        events.append({"key": (int(s), int(e)), "betas": g["beta"].to_numpy(),
                       "coords": g.select(cc).to_numpy(),
                       "mc": g["mc_index"].to_numpy()})
    events.sort(key=lambda ev: ev["key"])
    sel = [e for i, e in enumerate(events) if i % 2 == 1]
    con = [e for i, e in enumerate(events) if i % 2 == 0]
    print(f"{len(events)} events -> {len(sel)} select / {len(con)} confirm", flush=True)

    cfgs = []
    for tb, td in ((0.6, 0.1), (0.6, 0.05), (0.1, 0.05), (0.1, 0.1)):
        for tm in (0.0, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3):
            for ta in (0.0, 0.05):
                for mch in (0, 4):
                    cfgs.append((tb, td, tm, ta, mch))
    print(f"{len(cfgs)} configs", flush=True)

    G["events"] = sel
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_score, cfgs, chunksize=1)):
            rows.append(r)
            print(f"  [{i+1}/{len(cfgs)}] tb={r['tbeta']:.2f} td={r['td']:.2f} "
                  f"tm={r['merge_td']:.2f} ta={r['attach_td']:.2f} mch={r['mch']} "
                  f"-> fake {100*r['fake_rate']:6.2f}%  eff {100*r['efficiency']:6.2f}%",
                  flush=True)

    frontier = [r for r in rows if not any(
        o["efficiency"] > r["efficiency"] and o["fake_rate"] < r["fake_rate"]
        for o in rows)]
    frontier.sort(key=lambda r: r["efficiency"])
    print("\nPareto frontier (selection half):", flush=True)
    for r in frontier:
        print(f"  tb={r['tbeta']:.2f} td={r['td']:.2f} tm={r['merge_td']:.2f} "
              f"ta={r['attach_td']:.2f} mch={r['mch']}  "
              f"fake {100*r['fake_rate']:6.2f}%  eff {100*r['efficiency']:6.2f}%  "
              f"cand/ev {r['cand_per_ev']:.1f}", flush=True)

    best_eff = max(r["efficiency"] for r in rows)
    floor = min(0.90, best_eff - 0.01)
    best = min((r for r in rows if r["efficiency"] >= floor),
               key=lambda r: r["fake_rate"])
    G["events"] = con
    conf = _score((best["tbeta"], best["td"], best["merge_td"],
                   best["attach_td"], best["mch"]))
    print(f"\nwinner (select): tb={best['tbeta']} td={best['td']} "
          f"tm={best['merge_td']} ta={best['attach_td']} mch={best['mch']}  "
          f"fake {100*best['fake_rate']:.2f}%  eff {100*best['efficiency']:.2f}%", flush=True)
    print(f"winner (held-out half): fake {100*conf['fake_rate']:.2f}%  "
          f"eff {100*conf['efficiency']:.2f}%", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"grid": rows, "frontier": frontier,
                                 "best": best, "confirm": conf}, indent=2,
                                default=float) + "\n")


if __name__ == "__main__":
    main()
