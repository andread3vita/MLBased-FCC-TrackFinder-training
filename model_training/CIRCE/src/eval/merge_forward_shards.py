"""Validate and merge strided ``forward_pass.py`` cache shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def merge_frames(frames: list[pl.DataFrame], preserve_event_order: bool):
    """Concatenate shards without ever reordering hits inside an event."""
    combined = pl.concat(frames, how="vertical")
    if not preserve_event_order:
        combined = combined.sort(
            ["seed", "event_id"], maintain_order=True
        )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--preserve-event-order", action="store_true",
        help="keep shard/event row order for diagnostic position attachment",
    )
    args = parser.parse_args()
    if args.shard_count < 1:
        parser.error("--shard-count must be positive")

    root = Path(args.shard_root)
    manifests = []
    frames = []
    for index in range(args.shard_count):
        directory = root / f"shard_{index}"
        manifest_path = directory / "manifest.json"
        cache_path = directory / "forward_hits.parquet"
        if not manifest_path.is_file() or not cache_path.is_file():
            raise FileNotFoundError(f"incomplete forward shard {index}: {directory}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("event_shard_index") != index:
            raise ValueError(f"shard {index} has the wrong event_shard_index")
        if manifest.get("event_shard_count") != args.shard_count:
            raise ValueError(f"shard {index} has the wrong event_shard_count")
        manifests.append(manifest)
        frames.append(pl.read_parquet(cache_path))

    identity_fields = [
        "checkpoint_sha256", "seeds", "embed_dim", "max_hits",
        "min_signal_mc", "fix_particle_zero",
        "dataset_fingerprint", "inference_source_sha256",
        "drop_loopers", "merge_daughters", "algebra", "use_time",
        "pga_hit_encoding", "two_channel_dc", "cga_hit_encoding",
        "physical_drift_geometry", "separate_hit_metadata", "separate_hit_type",
        "layernorm_epsilon_mode",
        "normalize_mv_inputs", "fix_cga_null", "fix_wire_dir",
        "no_legacy_equivariance", "equivariance_group",
        "invariant_output_head",
    ]
    reference = manifests[0]
    for index, manifest in enumerate(manifests[1:], start=1):
        mismatched = [
            field for field in identity_fields
            if manifest.get(field) != reference.get(field)
        ]
        if mismatched:
            raise ValueError(
                f"shard {index} differs from shard 0 in {mismatched}"
            )

    combined = merge_frames(frames, args.preserve_event_order)
    min_signal_mc = reference.get("min_signal_mc")
    if min_signal_mc not in (0, 1):
        raise ValueError("shards do not record a valid min_signal_mc")
    particle_zero_rows = int((combined["mc_index"] == 0).sum())
    if (min_signal_mc == 0) != (particle_zero_rows > 0):
        raise ValueError(
            "actual particle-zero rows contradict shard truth policy"
        )
    duplicate_events = (
        combined.select(["seed", "event_id"])
        .unique()
        .height
    )
    expected_events = sum(int(item.get("n_events", 0)) for item in manifests)
    if duplicate_events != expected_events:
        raise ValueError(
            f"merged cache has {duplicate_events} unique events; "
            f"shard manifests claim {expected_events}"
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(output / "forward_hits.parquet", compression="zstd")
    merged_manifest = {
        **{field: reference.get(field) for field in identity_fields},
        "checkpoint": reference.get("checkpoint"),
        "merged_event_shards": args.shard_count,
        "n_events": expected_events,
        "n_hits": combined.height,
        "particle_zero_rows_verified": particle_zero_rows,
        "particle_zero_events_verified": combined.filter(
            pl.col("mc_index") == 0
        ).select(["seed", "event_id"]).unique().height,
    }
    merged_manifest["event_order_preserved"] = args.preserve_event_order
    (output / "manifest.json").write_text(
        json.dumps(merged_manifest, indent=2) + "\n"
    )
    print(
        f"merged {args.shard_count} shards: "
        f"{expected_events} events, {combined.height:,} hits"
    )


if __name__ == "__main__":
    main()
