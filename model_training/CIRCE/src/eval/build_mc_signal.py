"""Build a compact mc_particles table containing ONLY the signal particles
that appear in the forward-embedding cache.

The full mc_particles file is ~61M rows/seed (all Geant secondaries), so
loading 196 seeds is ~900 GB. The metric joins only need the signal
particles referenced by the clustered hits (~4.2M total). We filter one
seed at a time (peak ~5 GB) and write a small per-range part file.
"""
import argparse
import os
import time

import polars as pl

COLS = [
    "mc_index", "pt", "theta", "phi", "vx", "vy", "vz",
    "gen_status", "decayed_in_tracker", "charge", "pdg",
    "event_id", "seed",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="forward_hits.parquet (signal keys)")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--seed_start", type=int, required=True)
    ap.add_argument("--seed_end", type=int, required=True, help="inclusive")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    keys = (pl.read_parquet(args.cache, columns=["seed", "event_id", "mc_index"])
            .unique())
    parts = []
    t0 = time.time()
    for seed in range(args.seed_start, args.seed_end + 1):
        kk = keys.filter(pl.col("seed") == seed).select(["event_id", "mc_index"]).unique()
        if kk.height == 0:
            continue
        mc_path = os.path.join(args.data_dir, f"seed_{seed}", "mc_particles_train.parquet")
        if not os.path.exists(mc_path):
            print(f"  seed {seed}: MISSING mc file", flush=True)
            continue
        try:
            df = pl.read_parquet(mc_path, columns=COLS)
        except Exception as e:
            print(f"  seed {seed}: SKIP unreadable mc file ({type(e).__name__}: {e})", flush=True)
            continue
        sub = df.join(kk, on=["event_id", "mc_index"], how="semi")
        parts.append(sub)
        del df
        print(f"  seed {seed}: kept {sub.height:,} signal particles "
              f"(elapsed {time.time() - t0:.0f}s)", flush=True)

    out_df = pl.concat(parts, how="vertical_relaxed") if parts else pl.DataFrame(schema={c: pl.Float64 for c in COLS})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.write_parquet(args.out, compression="zstd")
    print(f"wrote {args.out}: {out_df.height:,} rows in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
