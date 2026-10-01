"""How much of v2_pzero's lead is the metric moving rather than the model improving?

`--fix_particle_zero` changes `noise_index` from 0 to -1, and the validation metric picks its
targets as `(~is_secondary) & (mc_index != noise_index)` (src/model.py:674). So the flag does two
things at once: it stops training against a contradictory label, and it enlarges the target set by
whatever `mc_index == 0` turns out to be. Only the first is a result; the second is a redefinition,
and the two must be separated before v2_pzero is compared with anything.

This counts the redefinition on the exact seeds the val metric runs on, so the split can be stated
rather than assumed.
"""

import glob
import os
import sys

import numpy as np
import pyarrow.parquet as pq

VAL_SEEDS = range(181, 191)
ROOT = os.environ.get("DATA", "/home/marko.cechovic/cgatr-data/data-final/parquet")


def main() -> int:
    n_ev = 0
    # The denominator of strict50 is not "every primary". `_compute_batch_metrics_greedy`
    # skips any true particle with fewer than 2 hits (src/model.py:708), so the count that
    # matters is over the >=2 hit set. Counting every primary overstates the denominator by
    # about 5x and makes the particle-0 redefinition look far smaller than it is.
    tgt_old = 0          # matchable, particle 0 called noise
    tgt_new = 0          # matchable, particle 0 called a track
    tgt_all_old = 0      # every primary, for reference against M23's density numbers
    tgt_ggtf = 0         # GGTF's own bar: >=3 hits
    p0_events = 0
    p0_hits = []
    all_hits = []

    cols = ["event_id", "mc_index", "produced_by_secondary"]
    for seed in VAL_SEEDS:
        seed_dir = os.path.join(ROOT, f"seed_{seed}")
        # The model is fed both hit collections, so a target has to be counted over both.
        parts = [
            pq.read_table(os.path.join(seed_dir, name), columns=cols)
            for name in ("dc_hits_train.parquet", "vtx_hits_train.parquet")
            if os.path.exists(os.path.join(seed_dir, name))
        ]
        if parts:
            ev = np.concatenate([p["event_id"].to_numpy() for p in parts])
            mc = np.concatenate([p["mc_index"].to_numpy() for p in parts])
            sec = np.concatenate(
                [p["produced_by_secondary"].to_numpy() for p in parts]
            ).astype(bool)

            order = np.argsort(ev, kind="stable")
            ev, mc, sec = ev[order], mc[order], sec[order]
            bounds = np.flatnonzero(np.diff(ev)) + 1

            for chunk_mc, chunk_sec in zip(np.split(mc, bounds), np.split(sec, bounds)):
                n_ev += 1
                prim = ~chunk_sec
                ids, counts = np.unique(chunk_mc[prim], return_counts=True)

                keep2 = counts >= 2
                tgt_all_old += int((ids != 0).sum())
                tgt_old += int(((ids != 0) & keep2).sum())
                tgt_new += int(keep2.sum())          # particle 0 is always >=2 hits when present
                tgt_ggtf += int(((ids != 0) & (counts >= 3)).sum())

                n0 = int(counts[ids == 0].sum()) if (ids == 0).any() else 0
                if n0:
                    p0_events += 1
                    p0_hits.append(n0)
                all_hits.extend(counts[ids != 0].tolist())

    if not n_ev:
        print(f"no events found under {ROOT}", file=sys.stderr)
        return 1

    p0_hits = np.array(p0_hits) if p0_hits else np.array([0])
    all_hits = np.array(all_hits) if all_hits else np.array([0])
    old_per_ev = tgt_old / n_ev
    new_per_ev = tgt_new / n_ev

    print(f"events                              {n_ev}")
    print(f"primaries/event, any hit count      {tgt_all_old / n_ev:.2f}   "
          f"(not the strict50 denominator)")
    print(f"strict50 denominator, noise=0       {old_per_ev:.2f}   "
          f">=2 hits, particle 0 called noise")
    print(f"strict50 denominator, noise=-1      {new_per_ev:.2f}   "
          f">=2 hits, particle 0 called a track")
    print(f"GGTF's target set, >=3 hits         {tgt_ggtf / n_ev:.2f}")
    print(f"particle 0 present in               {100 * p0_events / n_ev:.1f}% of events")
    print(f"particle 0 hits, median             {np.median(p0_hits):.0f}   "
          f"vs {np.median(all_hits):.0f} for an average primary")

    # If every added target were matched, strict50 moves from m/t to (m+d)/(t+d): the ceiling on
    # what the redefinition alone can be worth.
    print()
    for label, m in (("v2_a_nullfix", 0.362), ("v2_a_nullfix ep8", 0.433)):
        d = new_per_ev - old_per_ev
        ceiling = (m * old_per_ev + d) / new_per_ev
        print(f"redefinition ceiling on {label:20s} {100 * m:.1f}% -> {100 * ceiling:.1f}% "
              f"(+{100 * (ceiling - m):.1f} pt) if every particle 0 is matched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
