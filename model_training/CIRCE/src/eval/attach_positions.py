"""Recover hit positions for an existing forward cache, without a GPU.

The forward cache stores embedding coordinates, beta and mc_index, but not the
detector positions, so anything geometric -- helix-consistency merging above all
-- cannot run on it. Re-running the forward pass would cost GPU time we do not
have while the ablation arms are training.

It is recoverable instead, because the ordering is deterministic. The dataset
concatenates vertex hits then drift hits in parquet row order, the forward pass
keeps signal hits under the cache manifest's `min_signal_mc` policy, and
these caches were written with `max_hits = 0`, so nothing was subsampled. Rebuild
the same order from the raw parquet and the rows line up.

The alignment is not assumed: for every event the recovered mc_index sequence is
compared element-by-element against the cached one, and any mismatch is fatal.

Writes `forward_positions.parquet` beside the cache, same row order, so it can
be hstacked. `forward_pass.py` now stores positions directly, so this is only
needed for caches written before that.

    PYTHONPATH=. python src/eval/attach_positions.py \
        --cache_path eval_results/consol_r3_nokeepall/emb_all \
        --data_dir /home/marko.cechovic/cgatr-data/data-final/parquet
"""

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import polars as pl

POS_COLS = ["pos_x", "pos_y", "pos_z"]


def raw_signal_positions(
    data_dir, seed, event_id, grouped_cache, min_signal_mc
):
    """Positions of the signal hits of one event, in the dataset's own order."""
    key = (seed, "dc")
    if key not in grouped_cache:
        d = f"{data_dir}/seed_{seed}"
        dc = pl.read_parquet(
            f"{d}/dc_hits_train.parquet",
            columns=["hit_x", "hit_y", "hit_z", "mc_index",
                     "produced_by_secondary", "event_id"])
        vtx = pl.read_parquet(
            f"{d}/vtx_hits_train.parquet",
            columns=["hit_x", "hit_y", "hit_z", "mc_index",
                     "produced_by_secondary", "event_id"])
        grouped_cache.clear()  # one seed at a time keeps memory flat
        grouped_cache[(seed, "dc")] = {
            p["event_id"][0]: p for p in dc.partition_by("event_id")}
        grouped_cache[(seed, "vtx")] = {
            p["event_id"][0]: p for p in vtx.partition_by("event_id")}

    dc_df = grouped_cache[(seed, "dc")].get(event_id)
    vtx_df = grouped_cache[(seed, "vtx")].get(event_id)
    if dc_df is None or vtx_df is None:
        return None, None

    # vertex hits first, then drift hits -- the order _build_event_tensors uses
    both = pl.concat([vtx_df, dc_df])
    sig = both.filter(
        (pl.col("produced_by_secondary") == 0)
        & (pl.col("mc_index") >= min_signal_mc)
    )
    pos = sig.select(["hit_x", "hit_y", "hit_z"]).to_numpy().astype(np.float32)
    return pos, sig["mc_index"].to_numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache_path", required=True)
    ap.add_argument("--data_dir",
                    default="/home/marko.cechovic/cgatr-data/data-final/parquet")
    ap.add_argument("--out_name", default="forward_positions.parquet")
    args = ap.parse_args()

    t0 = time.time()
    cache_file = os.path.join(args.cache_path, "forward_hits.parquet")
    schema = pl.read_parquet_schema(cache_file)
    if "hit_order" not in schema:
        raise RuntimeError(
            "legacy cache has no unique hit_order key; repeated mc_index values "
            "cannot prove hit-level alignment. Regenerate with forward_pass.py."
        )
    hits = pl.read_parquet(
        cache_file,
        columns=["seed", "event_id", "hit_order", "mc_index"],
    )
    manifest_path = os.path.join(args.cache_path, "manifest.json")
    with open(manifest_path) as handle:
        cache_manifest = json.load(handle)
    min_signal_mc = int(cache_manifest.get("min_signal_mc", 1))
    print(f"cache: {hits.height:,} hits")

    groups = hits.group_by(["seed", "event_id"], maintain_order=True).agg(
        pl.col("hit_order"), pl.col("mc_index"))
    print(f"       {groups.height:,} events")

    grouped_cache = {}
    recovered_frames = []
    for i, row in enumerate(groups.iter_rows(named=True)):
        seed, eid = int(row["seed"]), int(row["event_id"])
        cached_order = np.asarray(row["hit_order"], dtype=np.int64)
        cached_mc = np.asarray(row["mc_index"], dtype=np.int64)
        pos, mc = raw_signal_positions(
            args.data_dir, seed, eid, grouped_cache, min_signal_mc
        )
        if (
            pos is None
            or len(mc) != len(cached_mc)
            or not np.array_equal(cached_order, np.arange(len(cached_order)))
            or not np.array_equal(mc, cached_mc)
        ):
            got = "missing" if pos is None else f"{len(mc)} hits"
            raise SystemExit(
                f"alignment failed at seed {seed} event {eid}: cache has "
                f"{len(cached_mc)} hits, raw gives {got}. The row order "
                f"assumption does not hold for this cache; regenerate it with "
                f"forward_pass.py, which now stores positions directly.")
        recovered_frames.append(pl.DataFrame({
            "seed": np.full(len(pos), seed, dtype=np.int64),
            "event_id": np.full(len(pos), eid, dtype=np.int64),
            "hit_order": np.arange(len(pos), dtype=np.int64),
            "raw_mc_index": mc.astype(np.int64),
            **{c: pos[:, k] for k, c in enumerate(POS_COLS)},
        }))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{groups.height} events  {time.time() - t0:.0f}s",
                  flush=True)

    recovered = pl.concat(recovered_frames, how="vertical")
    aligned = hits.join(
        recovered,
        on=["seed", "event_id", "hit_order"],
        how="left",
        validate="1:1",
        maintain_order="left",
    )
    if (
        len(aligned) != len(hits)
        or aligned["pos_x"].null_count()
        or not np.array_equal(
            aligned["mc_index"].to_numpy(),
            aligned["raw_mc_index"].to_numpy(),
        )
    ):
        raise RuntimeError("unique-key position alignment verification failed")
    df = aligned.select(POS_COLS)
    dest = os.path.join(args.cache_path, args.out_name)
    df.write_parquet(dest, compression="zstd")
    source_digest = hashlib.sha256()
    with open(cache_file, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            source_digest.update(chunk)
    positions_digest = hashlib.sha256()
    with open(dest, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            positions_digest.update(chunk)
    with open(
        os.path.join(args.cache_path, "forward_positions.manifest.json"), "w"
    ) as handle:
        json.dump({
            "source_forward_hits_sha256": source_digest.hexdigest(),
            "positions_sha256": positions_digest.hexdigest(),
            "rows": hits.height,
            "events": groups.height,
            "min_signal_mc": min_signal_mc,
            "oracle_warning": (
                "hit_x/y/z are truth-position diagnostics and cannot be "
                "promoted to deployable clustering"
            ),
        }, handle, indent=2)
    print(f"\nverified unique-key alignment on all {groups.height:,} events "
          f"({hits.height:,} hits)")
    print(f"wrote {dest} ({os.path.getsize(dest) / 1e6:.1f} MB) "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
