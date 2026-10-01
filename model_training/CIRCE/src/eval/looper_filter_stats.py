"""What GGTF's remove_loopers actually removes from our events.

The plan's open risk: our looper predicate is reconstructed from their code and
applied to our columns, so before any "matched regime" number is trusted the
resulting event sizes have to land near theirs. Their example IDEA event carries
694 nodes and 32 targets; ours have a median of about 3725 hits. If reproducing
their cut does not close most of that gap, either the predicate or the sample
differs and the comparison is not the one we think it is.

Reports, per event, hits and particles before and after, split by which clause
fired, and with daughter merging on and off.

    PYTHONPATH=. python src/eval/looper_filter_stats.py --seeds 181-190
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import polars as pl

from src.dataset.parquet_dataset import (
    _LOOPER_EXTENT_MM,
    _LOOPER_MIN_HITS,
    merge_daughter_particles,
)

# Their example IDEA event, quoted from the plan.
GGTF_NODES = 694
GGTF_TARGETS = 32


def event_extents(dc_df, vtx_df):
    """Per-particle bounding box and hit count, on GGTF's coordinates."""
    parts = []
    if len(dc_df):
        parts.append(dc_df.select(
            pl.col("mc_index"),
            pl.col("left_x").alias("x"),
            pl.col("left_y").alias("y"),
            pl.col("left_z").alias("z"),
        ))
    if len(vtx_df):
        parts.append(vtx_df.select(
            pl.col("mc_index"),
            pl.col("hit_x").alias("x"),
            pl.col("hit_y").alias("y"),
            pl.col("hit_z").alias("z"),
        ))
    if not parts:
        return None
    return (
        pl.concat(parts)
        .filter(pl.col("mc_index") > 0)
        .group_by("mc_index")
        .agg(
            (pl.col("x").max() - pl.col("x").min()).alias("ex"),
            (pl.col("y").max() - pl.col("y").min()).alias("ey"),
            (pl.col("z").max() - pl.col("z").min()).alias("ez"),
            pl.len().alias("n"),
        )
    )


def scan(data_dir, seeds, merge_daughters, max_events):
    lim_x, lim_y, lim_z = _LOOPER_EXTENT_MM
    rows = []
    n_done = 0
    for seed in seeds:
        d = Path(data_dir) / f"seed_{seed}"
        if not (d / "dc_hits_train.parquet").exists():
            continue
        dc_all = pl.read_parquet(d / "dc_hits_train.parquet")
        vtx_all = pl.read_parquet(d / "vtx_hits_train.parquet")
        mc_all = pl.read_parquet(d / "mc_particles_train.parquet")
        dc_g = {p["event_id"][0]: p for p in dc_all.partition_by("event_id")}
        vtx_g = {p["event_id"][0]: p for p in vtx_all.partition_by("event_id")}
        mc_g = {p["event_id"][0]: p for p in mc_all.partition_by("event_id")}

        for eid in sorted(set(dc_g) & set(vtx_g)):
            dc_df, vtx_df = dc_g[eid], vtx_g[eid]
            n_hits_0 = len(dc_df) + len(vtx_df)
            if merge_daughters:
                dc_df, vtx_df = merge_daughter_particles(
                    dc_df, vtx_df, mc_g.get(eid))
            agg = event_extents(dc_df, vtx_df)
            if agg is None or not len(agg):
                continue
            ex = agg["ex"].to_numpy()
            ey = agg["ey"].to_numpy()
            ez = agg["ez"].to_numpy()
            nh = agg["n"].to_numpy()

            big = (ex > lim_x) | (ey > lim_y) | (ez > lim_z)
            few = nh < _LOOPER_MIN_HITS
            drop = big | few
            rows.append({
                "hits_before": n_hits_0,
                "hits_after": int(nh[~drop].sum()),
                "hits_dropped_big": int(nh[big].sum()),
                "hits_dropped_few": int(nh[few & ~big].sum()),
                "part_before": len(nh),
                "part_after": int((~drop).sum()),
                "part_big": int(big.sum()),
                "part_few": int((few & ~big).sum()),
            })
            n_done += 1
            if n_done >= max_events:
                return pl.DataFrame(rows)
    return pl.DataFrame(rows)


def characterize(data_dir, seeds, max_events):
    """Per-particle table of what the cut keeps and what it throws away.

    The name remove_loopers implies the cut targets curlers, but a bounding box
    is a proxy for track LENGTH, not for how many times a track turns. A 0.1 GeV
    curler has a 167 mm bending radius, so it fits inside a 1600 mm box however
    many turns it makes, while a stiff track that crosses the chamber does not.
    So the cut plausibly removes the easy tracks and keeps the hard ones. This
    measures which, using the turning angle from the transverse momentum azimuth
    (see curler_diagnostic.py for why position azimuth will not do).
    """
    lim_x, lim_y, lim_z = _LOOPER_EXTENT_MM
    rows = []
    n_done = 0
    for seed in seeds:
        d = Path(data_dir) / f"seed_{seed}"
        if not (d / "dc_hits_train.parquet").exists():
            continue
        dc_all = pl.read_parquet(d / "dc_hits_train.parquet")
        vtx_all = pl.read_parquet(d / "vtx_hits_train.parquet")
        mc_all = pl.read_parquet(
            d / "mc_particles_train.parquet",
            columns=["mc_index", "event_id", "pt", "charge"])
        dc_g = {p["event_id"][0]: p for p in dc_all.partition_by("event_id")}
        vtx_g = {p["event_id"][0]: p for p in vtx_all.partition_by("event_id")}
        mc_g = {p["event_id"][0]: p for p in mc_all.partition_by("event_id")}

        for eid in sorted(set(dc_g) & set(vtx_g)):
            dc_df, vtx_df = dc_g[eid], vtx_g[eid]
            agg = event_extents(dc_df, vtx_df)
            if agg is None or not len(agg):
                continue
            # turning angle, from DC hits only (vertex hits are too few to help)
            turn = {}
            dcs = dc_df.filter(pl.col("mc_index") > 0)
            for (mc,), g in dcs.group_by(["mc_index"]):
                if len(g) < 3:
                    continue
                g = g.sort("time")
                psi = np.arctan2(g["hit_py"].to_numpy(), g["hit_px"].to_numpy())
                u = np.unwrap(psi)
                turn[int(mc)] = float(u.max() - u.min()) / (2 * np.pi)

            mc_df = mc_g.get(eid)
            pt_map = dict(zip(mc_df["mc_index"].to_list(),
                              mc_df["pt"].to_list())) if mc_df is not None else {}

            for r in agg.iter_rows(named=True):
                mc = int(r["mc_index"])
                big = (r["ex"] > lim_x) or (r["ey"] > lim_y) or (r["ez"] > lim_z)
                few = r["n"] < _LOOPER_MIN_HITS
                rows.append({
                    "n_hits": int(r["n"]),
                    "pt": float(pt_map.get(mc, float("nan"))),
                    "n_loops": turn.get(mc, 0.0),
                    "dropped": bool(big or few),
                    "big": bool(big),
                })
            n_done += 1
            if n_done >= max_events:
                return pl.DataFrame(rows)
    return pl.DataFrame(rows)


def report_characterize(df):
    def line(name, sub):
        if not len(sub):
            print(f"  {name:<22}      0")
            return
        pt = sub["pt"].to_numpy()
        pt = pt[np.isfinite(pt)]
        lo = sub["n_loops"].to_numpy()
        print(f"  {name:<22} {len(sub):6d}  "
              f"hits {sub['n_hits'].median():6.0f}  "
              f"pT {np.median(pt) if len(pt) else float('nan'):7.3f}  "
              f"turns {np.median(lo):6.2f}  "
              f">1 turn {100 * (lo > 1).mean():5.1f}%  "
              f">2 turns {100 * (lo > 2).mean():5.1f}%")

    print("=== what the cut keeps and drops, per particle ===")
    print(f"  {'population':<22} {'n':>6}  {'med hits':>4}  {'med pT':>9}  "
          f"{'med turns':>6}")
    line("kept", df.filter(~pl.col("dropped")))
    line("dropped, too extended", df.filter(pl.col("big")))
    line("dropped, < 5 hits", df.filter(pl.col("dropped") & ~pl.col("big")))
    print()
    # A particle with >2 full turns is unambiguously a curler.
    curl = df.filter(pl.col("n_loops") > 2)
    if len(curl):
        kept = 100 * (~curl["dropped"]).mean()
        print(f"  of {len(curl)} particles turning more than twice, "
              f"{kept:.1f}% SURVIVE the cut")
    stiff = df.filter(pl.col("pt") > 1.0)
    if len(stiff):
        print(f"  of {len(stiff)} particles with pT > 1 GeV, "
              f"{100 * stiff['dropped'].mean():.1f}% are removed")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir",
                    default="/home/marko.cechovic/cgatr-data/data-final/parquet")
    ap.add_argument("--seeds", default="181-190")
    ap.add_argument("--max_events", type=int, default=500)
    args = ap.parse_args()

    a, b = args.seeds.split("-")
    seeds = range(int(a), int(b) + 1)

    print(__doc__)
    print(f"thresholds: extent > {_LOOPER_EXTENT_MM} mm  or  hits < "
          f"{_LOOPER_MIN_HITS}\n")

    for merge in (False, True):
        df = scan(args.data_dir, seeds, merge, args.max_events)
        if not len(df):
            print("no events found")
            return
        tag = "merge_daughters ON " if merge else "merge_daughters OFF"
        m = {c: float(df[c].mean()) for c in df.columns}
        keep_h = 100 * m["hits_after"] / m["hits_before"]
        keep_p = 100 * m["part_after"] / m["part_before"]
        print(f"=== {tag} ({len(df)} events) ===")
        print(f"  hits/event       {m['hits_before']:8.1f} -> {m['hits_after']:8.1f}"
              f"   ({keep_h:.1f}% kept)")
        print(f"    dropped as too extended {m['hits_dropped_big']:8.1f}")
        print(f"    dropped as < {_LOOPER_MIN_HITS} hits      "
              f"{m['hits_dropped_few']:8.1f}")
        print(f"  particles/event  {m['part_before']:8.1f} -> {m['part_after']:8.1f}"
              f"   ({keep_p:.1f}% kept)")
        print(f"    too extended            {m['part_big']:8.1f}")
        print(f"    < {_LOOPER_MIN_HITS} hits                 {m['part_few']:8.1f}")
        print(f"  GGTF reference   {GGTF_NODES} nodes, {GGTF_TARGETS} targets  "
              f"-> ratio hits {m['hits_after'] / GGTF_NODES:.2f}x, "
              f"targets {m['part_after'] / GGTF_TARGETS:.2f}x")
        print()

    q = np.percentile(df["hits_after"].to_numpy(), [5, 50, 95])
    print(f"filtered hits/event percentiles (5/50/95): "
          f"{q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f}\n")

    report_characterize(
        characterize(args.data_dir, seeds, min(args.max_events, 120)))


if __name__ == "__main__":
    main()
