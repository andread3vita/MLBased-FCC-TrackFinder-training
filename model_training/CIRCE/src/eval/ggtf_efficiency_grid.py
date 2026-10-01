"""The 2x2 trackingEfficiencyPlot grid (genStatus x nHits) as numbers.

Same matching as ggtf_exact_fake_rate (purity > 0.75), same denominators as
their notebook call sites: genStatus [1] or [0,1], nHits > 3 or > 10, at
15 < theta < 165 and pT > 0. Prints the grid and writes a JSON next to it.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/home/marko.cechovic/cgatr-paper/literature/cgatr_fcc_pkg")
import numpy as np
import polars as pl

from src.eval.fcc_cache_parallel import cluster_event as apply_clusterer
from src.eval.ggtf_clustering_sweep import load_events

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--dir", required=True,
                help="eval dir with emb/forward_hits.parquet + mc_signal.parquet")
ap.add_argument("--tbeta", type=float, default=0.6)
ap.add_argument("--td", type=float, default=0.3)
ap.add_argument("--out", type=Path)
args = ap.parse_args()

events = load_events(Path(f"{args.dir}/emb/forward_hits.parquet"), 4)
mc = pl.read_parquet(f"{args.dir}/mc_signal.parquet",
                     columns=["mc_index", "event_id", "seed", "gen_status", "pt", "theta"])
truth = {(int(r["seed"]), int(r["event_id"]), int(r["mc_index"])):
         (r["gen_status"], r["pt"], np.degrees(r["theta"]))
         for r in mc.iter_rows(named=True)}

rows = []
for ev in events:
    labels = np.asarray(apply_clusterer(
        "greedy", ev["betas"], ev["coords"], args.tbeta, args.td, 0))
    arr = ev["mc"]
    uniq, counts = np.unique(arr, return_counts=True)
    nhits = {int(p): int(c) for p, c in zip(uniq, counts)}
    targets = {p for p, c in nhits.items() if c >= 3}
    csize = defaultdict(int)
    for lab in labels[labels >= 0]:
        csize[int(lab)] += 1
    overlap = defaultdict(int)
    sel = labels >= 0
    for p, lab in zip(arr[sel], labels[sel]):
        if int(p) in targets:
            overlap[(int(p), int(lab))] += 1
    matched = {p for (p, lab), n in overlap.items() if n / csize[lab] > 0.75}
    for p in targets:
        gs, pt, th = truth.get((ev["seed"], ev["event_id"], p), (None, None, None))
        if gs is None or pt is None or not pt > 0 or not 15 < th < 165:
            continue
        rows.append((nhits[p], gs, pt, p in matched))

grid = {}
print(f"{len(events)} events, tbeta/td = {args.tbeta:g}/{args.td:g}, 15 < theta < 165:")
for gens in ((1,), (0, 1)):
    for nmin in (3, 10):
        sub = [m for nh, gs, pt, m in rows if gs in gens and nh > nmin]
        key = f"genStatus={list(gens)},nHits>{nmin}"
        grid[key] = {"efficiency": float(np.mean(sub)) if sub else None,
                     "n_targets": len(sub)}
        print(f"  {key:>26}: {100*np.mean(sub):6.2f}%  ({len(sub):,} targets)")
if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"tbeta": args.tbeta, "td": args.td, "n_events": len(events), "grid": grid},
        indent=2) + "\n")
