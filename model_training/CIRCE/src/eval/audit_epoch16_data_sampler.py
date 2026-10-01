"""Replay the final sampler and audit no-keep-all target/join semantics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import ks_2samp

from src.dataset.parquet_dataset import IDEAParquetDataset, TokenBudgetBatchSampler


def _describe(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "minimum": float(np.min(values)),
        "q10": float(np.quantile(values, 0.1)),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.9)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _index_seed(entry) -> int:
    return int(Path(entry[0]).parent.name.replace("seed_", ""))


def _physics_rows(dataset, indices):
    by_seed = defaultdict(set)
    for index in indices:
        entry = dataset._index[index]
        by_seed[_index_seed(entry)].add(int(entry[2]))

    rows = []
    duplicate_mc_rows = 0
    missing_hit_particle_keys = 0
    for seed, event_ids in sorted(by_seed.items()):
        seed_dir = Path(dataset._index[next(
            index for index in indices
            if _index_seed(dataset._index[index]) == seed
        )][0]).parent
        event_filter = pl.col("event_id").is_in(sorted(event_ids))
        hit_keys = pl.concat([
            pl.scan_parquet(seed_dir / "dc_hits_train.parquet")
            .filter(event_filter)
            .select(["event_id", "mc_index"]),
            pl.scan_parquet(seed_dir / "vtx_hits_train.parquet")
            .filter(event_filter)
            .select(["event_id", "mc_index"]),
        ]).group_by(["event_id", "mc_index"]).agg(
            pl.len().alias("n_hits")
        ).collect()
        particles = (
            pl.scan_parquet(seed_dir / "mc_particles_train.parquet")
            .filter(event_filter)
            .select(["event_id", "mc_index", "pt"])
            .collect()
        )
        duplicate_mc_rows += (
            particles.group_by(["event_id", "mc_index"]).len()
            .filter(pl.col("len") != 1).height
        )
        missing_hit_particle_keys += hit_keys.join(
            particles.select(["event_id", "mc_index"]),
            on=["event_id", "mc_index"], how="anti",
        ).height
        joined = hit_keys.join(
            particles, on=["event_id", "mc_index"], how="left", validate="1:1"
        ).with_columns(pl.lit(seed).alias("seed"))
        rows.append(joined)
    return pl.concat(rows), duplicate_mc_rows, missing_hit_particle_keys


def _physics_summary(rows: pl.DataFrame, n_events: int) -> dict:
    targets = rows.filter(pl.col("n_hits") >= 3)
    particle_zero = rows.filter(pl.col("mc_index") == 0)
    zero_targets = particle_zero.filter(pl.col("n_hits") >= 3)
    pt = targets["pt"].drop_nulls().to_numpy()
    return {
        "events": n_events,
        "particles_with_hits": len(rows),
        "targets_min_3_hits": len(targets),
        "targets_per_event": len(targets) / max(n_events, 1),
        "exactly_3_hit_targets": int((targets["n_hits"] == 3).sum()),
        "exactly_3_hit_target_fraction": float(
            (targets["n_hits"] == 3).mean()
        ),
        "particle_zero_rows": len(particle_zero),
        "particle_zero_targets": len(zero_targets),
        "particle_zero_target_fraction": (
            len(zero_targets) / max(len(targets), 1)
        ),
        "pt": {
            **_describe(pt),
            "fraction_0p1_0p2": float(np.mean((pt >= 0.1) & (pt < 0.2))),
            "fraction_0p2_0p5": float(np.mean((pt >= 0.2) & (pt < 0.5))),
            "fraction_below_0p1": float(np.mean(pt < 0.1)),
        },
    }


def _first_rank_batches(
    dataset, count=40, *, world_size=4, drop_last=False,
    stable_epoch_length=False,
):
    sampler = TokenBudgetBatchSampler(
        dataset, max_tokens=16000, shuffle=True, drop_last=drop_last,
        stable_epoch_length=stable_epoch_length, verbose=False,
    )
    batches = sampler._pack_global_batches()
    n_keep = (len(batches) // world_size) * world_size
    if stable_epoch_length:
        n_keep = min(n_keep, sampler._stable_global_target(world_size))
    batches = batches[:n_keep]
    rank_batches = [batches[rank::world_size] for rank in range(world_size)]
    selected = [rank[:count] for rank in rank_batches]
    indices = [
        index
        for rank in selected
        for batch in rank
        for index in batch
    ]
    return rank_batches, selected, indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = IDEAParquetDataset(
        args.data_dir, seed_range=(1, 181), min_target_hits=3
    )
    validation = IDEAParquetDataset(
        args.data_dir, seed_range=(181, 191), min_target_hits=3
    )
    val_all_rank_batches, val_batches, val_indices = _first_rank_batches(
        validation
    )
    _, replay_batches, replay_indices = _first_rank_batches(validation)
    _, train_batches, train_indices = _first_rank_batches(
        train, drop_last=True, stable_epoch_length=True
    )

    val_rows, val_duplicates, val_missing = _physics_rows(
        validation, range(len(validation))
    )
    fixed_rows, fixed_duplicates, fixed_missing = _physics_rows(
        validation, val_indices
    )
    train_rows, train_duplicates, train_missing = _physics_rows(
        train, train_indices
    )
    train_sizes = [entry[3] for entry in train._index]
    val_sizes = [entry[3] for entry in validation._index]
    fixed_sizes = [validation._index[index][3] for index in val_indices]
    fixed_train_sizes = [train._index[index][3] for index in train_indices]

    report = {
        "scope": (
            "configured no-keep-all dataset only: train seeds 1-180 and "
            "validation seeds 181-190; the script accepts no holdout path"
        ),
        "sampler": {
            "max_tokens": 16000,
            "world_size": 4,
            "limit_val_batches_per_rank": 40,
            "fixed_replay_identical": (
                val_batches == replay_batches and val_indices == replay_indices
            ),
            "validation_batches_per_rank": [
                len(rank) for rank in val_all_rank_batches
            ],
            "fixed_validation_rank_batches": sum(
                len(rank) for rank in val_batches
            ),
            "fixed_validation_events": len(val_indices),
            "fixed_train_rank_batches": sum(
                len(rank) for rank in train_batches
            ),
            "fixed_train_events": len(train_indices),
        },
        "event_hit_counts": {
            "full_train": _describe(train_sizes),
            "fixed_train_40_batches": _describe(fixed_train_sizes),
            "full_validation": _describe(val_sizes),
            "fixed_validation_40_batches": _describe(fixed_sizes),
            "ks_fixed_validation_vs_full_validation": {
                "statistic": float(ks_2samp(fixed_sizes, val_sizes).statistic),
                "pvalue": float(ks_2samp(fixed_sizes, val_sizes).pvalue),
            },
            "ks_fixed_validation_vs_fixed_train": {
                "statistic": float(
                    ks_2samp(fixed_sizes, fixed_train_sizes).statistic
                ),
                "pvalue": float(
                    ks_2samp(fixed_sizes, fixed_train_sizes).pvalue
                ),
            },
        },
        "physics": {
            "full_validation": _physics_summary(
                val_rows, len(validation)
            ),
            "fixed_validation_40_batches": _physics_summary(
                fixed_rows, len(val_indices)
            ),
            "fixed_train_40_batches": _physics_summary(
                train_rows, len(train_indices)
            ),
        },
        "joins": {
            "full_validation_duplicate_mc_keys": val_duplicates,
            "full_validation_hit_particle_keys_missing_mc": val_missing,
            "fixed_validation_duplicate_mc_keys": fixed_duplicates,
            "fixed_validation_hit_particle_keys_missing_mc": fixed_missing,
            "fixed_train_duplicate_mc_keys": train_duplicates,
            "fixed_train_hit_particle_keys_missing_mc": train_missing,
        },
        "semantic_mismatch": {
            "training_target_min_hits": 3,
            "paper_candidate_min_hits": 4,
            "affected_target_proxy": (
                "fraction of truth targets with exactly three hits; candidate "
                "post-cut impact is measured separately in the cluster sweep"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
