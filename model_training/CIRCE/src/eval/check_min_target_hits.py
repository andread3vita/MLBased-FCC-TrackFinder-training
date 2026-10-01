"""Does `--min_target_hits 3` actually reproduce GGTF's target set?

Two things have to hold before the parity re-run is worth 60 hours of GPU:

1. `relabel_small_targets` agrees hit-for-hit with a naive per-event loop. The tensor version
   keys on (event, particle) pairs through `torch.unique(dim=0)`, and getting that wrong in a
   way that only shows up across event boundaries would silently corrupt every target.
2. The surviving target count matches the 33.2/event that GGTF's rule produces on our data,
   and the hits are all still present -- relabelled, not deleted (M23).
"""

import glob
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.lightning_module import relabel_small_targets  # noqa: E402

ROOT = os.environ.get("DATA", "/home/marko.cechovic/cgatr-data/data-final/parquet")
SEEDS = range(181, 191)
MIN_HITS = int(os.environ.get("MIN_HITS", "3"))
NOISE = int(os.environ.get("NOISE", "-1"))


def naive(mc, ev, noise_index, min_hits):
    """The rule as a person would write it, one event at a time."""
    out = mc.copy()
    for e in np.unique(ev):
        sel = np.flatnonzero(ev == e)
        vals, counts = np.unique(mc[sel], return_counts=True)
        small = set(vals[counts < min_hits].tolist()) - {noise_index}
        for i in sel:
            if mc[i] in small:
                out[i] = noise_index
    return out


def main() -> int:
    cols = ["event_id", "mc_index", "produced_by_secondary"]
    ev_all, mc_all = [], []
    for seed in SEEDS:
        d = os.path.join(ROOT, f"seed_{seed}")
        for name in ("dc_hits_train.parquet", "vtx_hits_train.parquet"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                t = pq.read_table(p, columns=cols)
                # event_id restarts in every seed, so an event is (seed, event_id). Keying on
                # event_id alone silently merges ten events into one and inflates every count.
                ev_all.append(seed * 1_000_000 + t["event_id"].to_numpy().astype(np.int64))
                mc_all.append(t["mc_index"].to_numpy())

    ev = np.concatenate(ev_all)
    mc = np.concatenate(mc_all)
    order = np.argsort(ev, kind="stable")
    ev, mc = ev[order], mc[order]

    _, ev = np.unique(ev, return_inverse=True)
    n_ev = int(ev.max()) + 1

    # Agreement is checked on a slice small enough for the O(n^2)-ish naive loop.
    probe = ev < 200
    got = relabel_small_targets(
        torch.from_numpy(mc[probe].copy()), torch.from_numpy(ev[probe]), NOISE, MIN_HITS
    ).numpy()
    want = naive(mc[probe], ev[probe], NOISE, MIN_HITS)
    if not np.array_equal(got, want):
        bad = int((got != want).sum())
        print(f"FAIL: tensor and naive disagree on {bad} of {probe.sum()} hits")
        return 1
    print(f"tensor implementation agrees with the naive loop on {int(probe.sum())} hits, "
          f"200 events")

    full = relabel_small_targets(
        torch.from_numpy(mc.copy()), torch.from_numpy(ev), NOISE, MIN_HITS
    ).numpy()

    # ev is sorted, so events are contiguous and a split beats masking 20M rows per event.
    bounds = np.flatnonzero(np.diff(ev)) + 1
    before = sum(np.unique(c).size for c in np.split(mc, bounds))
    after = sum(int((np.unique(c) != NOISE).sum()) for c in np.split(full, bounds))
    scale = n_ev

    print(f"hits                       {len(mc)} before, {len(full)} after   "
          f"(must be equal: relabel, not delete)")
    print(f"targets/event              {before / scale:.2f} -> {after / scale:.2f}")
    print(f"hits now labelled noise    {100 * (full == NOISE).mean():.1f}%")

    ok = len(mc) == len(full) and abs(after / scale - 33.4) < 1.5
    print("\n" + ("PASS: matches GGTF's ~33 targets/event and keeps every hit"
                  if ok else "FAIL: target count is not in GGTF's range"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
