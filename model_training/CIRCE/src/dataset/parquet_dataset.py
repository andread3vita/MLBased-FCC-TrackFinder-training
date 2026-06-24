"""Polars-based Parquet dataset for IDEA detector CGA track finding.

Lazy-loading: only stores lightweight metadata at init.
Tensors are built on-demand in __getitem__ with an LRU cache for parquet reads.
"""

import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, Sampler

try:
    import polars as pl
except ImportError:
    raise ImportError("Install polars: pip install polars")


def _read_and_group(parquet_path: str):
    """Read a parquet file and return a dict of {event_id: DataFrame}."""
    df = pl.read_parquet(parquet_path)
    grouped = {}
    for part in df.partition_by("event_id"):
        grouped[part["event_id"][0]] = part
    return grouped


# Module-level LRU cache — shared across dataset instances within one process.
# Keyed by file path string; caches the grouped dict so repeated __getitem__
# calls for events in the same seed don't re-read parquet.
_PARQUET_CACHE_SIZE = int(os.environ.get("CGATR_PARQUET_CACHE_SIZE", "64"))

@lru_cache(maxsize=_PARQUET_CACHE_SIZE)
def _cached_read(parquet_path: str):
    return _read_and_group(parquet_path)


def _build_event_tensors(dc_df, vtx_df):
    """Convert Polars DataFrames for one event into feature/label tensors."""
    n_dc = len(dc_df)
    n_vtx = len(vtx_df)
    n_total = n_dc + n_vtx

    vtx_pos = torch.tensor(
        vtx_df.select(["hit_x", "hit_y", "hit_z"]).to_numpy(), dtype=torch.float32
    )
    dc_wire = torch.tensor(
        dc_df.select([
            "wire_x", "wire_y", "wire_z",
            "drift_distance", "wire_azimuthal_angle", "wire_stereo_angle"
        ]).to_numpy(), dtype=torch.float32
    )
    dc_pos = torch.tensor(
        dc_df.select(["hit_x", "hit_y", "hit_z"]).to_numpy(), dtype=torch.float32
    )
    vtx_mc = torch.tensor(vtx_df["mc_index"].to_numpy(), dtype=torch.long)
    dc_mc = torch.tensor(dc_df["mc_index"].to_numpy(), dtype=torch.long)
    vtx_sec = torch.tensor(vtx_df["produced_by_secondary"].to_numpy(), dtype=torch.bool)
    dc_sec = torch.tensor(dc_df["produced_by_secondary"].to_numpy(), dtype=torch.bool)

    features = torch.zeros(n_total, 10, dtype=torch.float32)
    features[:n_vtx, :3] = vtx_pos
    features[:n_vtx, 3] = 0.0
    features[n_vtx:, :3] = dc_pos
    features[n_vtx:, 3] = 1.0
    features[n_vtx:, 4:] = dc_wire

    return {
        "features": features,
        "mc_index": torch.cat([vtx_mc, dc_mc], dim=0),
        "is_secondary": torch.cat([vtx_sec, dc_sec], dim=0),
        "n_hits": n_total,
        "n_vtx": n_vtx,
        "n_dc": n_dc,
    }


class IDEAParquetDataset(Dataset):
    """Dataset that loads IDEA detector hits from Parquet files.

    Lazy-loading: __init__ only scans for valid (seed, event_id) pairs.
    Tensors are built on-demand in __getitem__.

    Expected directory structure:
        data_dir/
            seed_1/
                dc_hits_train.parquet
                vtx_hits_train.parquet
    """

    def __init__(
        self,
        data_dir: str,
        seed_range: Tuple[int, int] = None,
        seed_list=None,
        max_hits_per_event: Optional[int] = None,
    ):
        self.max_hits = max_hits_per_event
        # Store lightweight metadata: (dc_path, vtx_path, event_id, n_total)
        self._index = []

        data_dir = Path(data_dir)

        if seed_list is not None:
            seed_indices = seed_list
        elif seed_range is not None:
            seed_indices = list(range(seed_range[0], seed_range[1]))
        else:
            seed_indices = list(range(1, 601))

        for seed_idx in seed_indices:
            seed_dir = data_dir / f"seed_{seed_idx}"
            if not seed_dir.exists():
                continue

            dc_path = seed_dir / "dc_hits_train.parquet"
            vtx_path = seed_dir / "vtx_hits_train.parquet"

            if not dc_path.exists() or not vtx_path.exists():
                continue

            try:
                # Lightweight scan: only read event_id + row counts
                dc_counts = (
                    pl.scan_parquet(str(dc_path))
                    .group_by("event_id")
                    .agg(pl.len().alias("n"))
                    .collect()
                )
                vtx_counts = (
                    pl.scan_parquet(str(vtx_path))
                    .group_by("event_id")
                    .agg(pl.len().alias("n"))
                    .collect()
                )

                dc_map = dict(zip(
                    dc_counts["event_id"].to_list(),
                    dc_counts["n"].to_list(),
                ))
                vtx_map = dict(zip(
                    vtx_counts["event_id"].to_list(),
                    vtx_counts["n"].to_list(),
                ))

                common_ids = sorted(set(dc_map) & set(vtx_map))
                dc_str = str(dc_path)
                vtx_str = str(vtx_path)

                for eid in common_ids:
                    n_total = dc_map[eid] + vtx_map[eid]
                    # Store true size for sampler budgeting; truncation is in __getitem__
                    effective = min(n_total, max_hits_per_event) if max_hits_per_event else n_total
                    self._index.append((dc_str, vtx_str, eid, effective))

            except Exception as e:
                print(f"Warning: could not scan seed {seed_idx}: {e}")
                continue

        if self._index:
            sizes = [e[3] for e in self._index]
            sizes.sort()
            print(f"IDEAParquetDataset: {len(self._index)} events from {len(seed_indices)} seeds")
            print(f"  Hits/event: min={sizes[0]}, median={sizes[len(sizes)//2]}, "
                  f"max={sizes[-1]}, total={sum(sizes):,}")
        else:
            print(f"IDEAParquetDataset: 0 events from {len(seed_indices)} seeds")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        dc_path, vtx_path, eid, _ = self._index[idx]

        dc_grouped = _cached_read(dc_path)
        vtx_grouped = _cached_read(vtx_path)

        dc_df = dc_grouped.get(eid)
        vtx_df = vtx_grouped.get(eid)
        if dc_df is None or vtx_df is None:
            return None

        # Subsample oversized events to keep batches tractable.
        # NOTE: treat 0 (and None) as UNCAPPED. Previously `is not None` made
        # max_hits=0 truncate every event to 1 DC hit (max(0 - n_vtx, 1)).
        if self.max_hits:
            n_total = len(dc_df) + len(vtx_df)
            if n_total > self.max_hits:
                # Keep all VTX hits (few), subsample DC hits
                keep_dc = max(self.max_hits - len(vtx_df), 1)
                if keep_dc < len(dc_df):
                    dc_df = dc_df.sample(keep_dc, seed=idx)

        return _build_event_tensors(dc_df, vtx_df)


def collate_idea_events(batch):
    """Custom collate: concatenates events and returns seq_lens for attention mask."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    features = torch.cat([b["features"] for b in batch], dim=0)
    mc_index = torch.cat([b["mc_index"] for b in batch], dim=0)
    is_secondary = torch.cat([b["is_secondary"] for b in batch], dim=0)
    seq_lens = [b["n_hits"] for b in batch]

    return {
        "features": features,
        "mc_index": mc_index,
        "is_secondary": is_secondary,
        "seq_lens": seq_lens,
    }


class TokenBudgetBatchSampler(Sampler):
    """Batch sampler that packs events up to a token (hit) budget.

    Uses pre-computed n_total from IDEAParquetDataset._index to form
    batches whose total hit count stays under max_tokens.

    DDP safety contract (v37 hardening):
      1. The batch list is built once per epoch in `set_epoch()` and cached, so
         `__len__` and `__iter__` agree byte-for-byte and the LR scheduler gets
         the correct step count.
      2. The global batch list is *truncated* to a multiple of world_size
         BEFORE the per-rank slice, so every rank yields the exact same
         number of batches. This eliminates the "one-rank-finishes-first"
         class of NCCL deadlocks.
      3. Events whose hit count exceeds `max_tokens` are admitted as singleton
         batches by default (they exceed the budget but at least train on the
         data). Pass `drop_oversized=True` to filter them, or raise
         `--max_tokens` to cover the full distribution.
      4. RNG seeded by epoch only — calling `__iter__` does NOT advance the
         seed, so DataLoader prefetching / multiple iterations within one epoch
         remain deterministic.
    """

    def __init__(self, dataset, max_tokens: int, shuffle: bool = True,
                 drop_last: bool = True, drop_oversized: bool = False,
                 verbose: bool = True):
        self.sizes = [entry[3] for entry in dataset._index]
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.drop_oversized = drop_oversized
        self.verbose = verbose
        self._epoch = 0
        self._cached_batches = None

        # Filter oversized events once. Singletons exceeding max_tokens are the
        # main OOM risk under AMP + grad_checkpoint, and we cannot honour the
        # token budget for them anyway.
        if drop_oversized:
            n_oversized = sum(1 for s in self.sizes if s > max_tokens)
            if n_oversized > 0 and verbose:
                rank = int(os.environ.get("LOCAL_RANK", 0))
                if rank == 0:
                    largest = max(self.sizes)
                    print(
                        f"  TokenBudgetBatchSampler: dropping {n_oversized}/"
                        f"{len(self.sizes)} events with hits > max_tokens="
                        f"{max_tokens} (largest={largest}). Increase "
                        f"--max_tokens to keep them.",
                        flush=True,
                    )

    def _get_rank_info(self):
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        return rank, world_size

    def _build_batches(self):
        """Build the canonical (rank-sliced, equal-length) batch list for the
        current epoch. Called lazily by both __iter__ and __len__ so they
        agree."""
        rank, world_size = self._get_rank_info()
        rng = random.Random(42 + self._epoch)

        indices = [
            i for i, s in enumerate(self.sizes)
            if (not self.drop_oversized) or s <= self.max_tokens
        ]
        # Sort by size so similar events group together (better GPU util).
        indices.sort(key=lambda i: self.sizes[i])

        if self.shuffle:
            bucket_size = 80
            buckets = [indices[i:i + bucket_size]
                       for i in range(0, len(indices), bucket_size)]
            rng.shuffle(buckets)
            for bucket in buckets:
                rng.shuffle(bucket)
            indices = [idx for bucket in buckets for idx in bucket]

        # Pack into batches respecting token budget.
        batches = []
        current_batch = []
        current_tokens = 0
        for idx in indices:
            event_size = self.sizes[idx]
            if current_batch and current_tokens + event_size > self.max_tokens:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(idx)
            current_tokens += event_size
        if current_batch:
            if not self.drop_last or len(batches) == 0:
                batches.append(current_batch)

        if self.shuffle:
            rng.shuffle(batches)

        # DDP: truncate to multiple of world_size BEFORE per-rank slice so all
        # ranks yield the exact same number of batches. This is the key
        # invariant that prevents NCCL collective desync.
        if world_size > 1:
            n_keep = (len(batches) // world_size) * world_size
            batches = batches[:n_keep]
            batches = batches[rank::world_size]
        return batches

    def set_epoch(self, epoch: int):
        """Set epoch and rebuild the cached batch list."""
        self._epoch = epoch
        self._cached_batches = self._build_batches()

    def __iter__(self):
        if self._cached_batches is None:
            self._cached_batches = self._build_batches()
        yield from self._cached_batches

    def __len__(self):
        if self._cached_batches is None:
            self._cached_batches = self._build_batches()
        return len(self._cached_batches)
