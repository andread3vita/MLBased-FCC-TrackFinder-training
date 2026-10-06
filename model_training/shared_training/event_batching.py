"""Canonical event batching shared by the matched CIRCE and GATr trainers.

The token-budget implementation is CIRCE's production sampler.  It builds one
global event/batch plan and only then divides complete batches between DDP
ranks.  DataLoader workers therefore load planned events; they never shard
files or independently repack the stream.
"""

from __future__ import annotations

import os
import random
from typing import Optional

from torch.utils.data import Sampler


class TokenBudgetBatchSampler(Sampler):
    """Pack indexed events up to a hit budget using CIRCE's exact policy."""

    def __init__(
        self,
        dataset,
        max_tokens: int,
        shuffle: bool = True,
        drop_last: bool = True,
        drop_oversized: bool = False,
        verbose: bool = True,
        stable_epoch_length: bool = True,
        probe_epochs: int = 64,
        seed: int = 42,
    ):
        self.sizes = [entry[3] for entry in dataset._index]
        self.max_tokens = int(max_tokens)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.drop_oversized = bool(drop_oversized)
        self.verbose = bool(verbose)
        self.stable_epoch_length = bool(stable_epoch_length)
        self.probe_epochs = int(probe_epochs)
        self.seed = int(seed)
        self._epoch = 0
        self._cached_batches = None
        self._fixed_global: Optional[int] = None

        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.drop_oversized:
            n_oversized = sum(size > self.max_tokens for size in self.sizes)
            if n_oversized and self.verbose and self._get_rank_info()[0] == 0:
                print(
                    "  TokenBudgetBatchSampler: dropping "
                    f"{n_oversized}/{len(self.sizes)} events with hits > "
                    f"max_tokens={self.max_tokens} (largest={max(self.sizes)}). "
                    "Increase --max_tokens to keep them.",
                    flush=True,
                )

    @staticmethod
    def _get_rank_info():
        return (
            int(os.environ.get("LOCAL_RANK", 0)),
            int(os.environ.get("WORLD_SIZE", 1)),
        )

    def _stable_global_target(self, world_size: int) -> int:
        if self._fixed_global is not None:
            return self._fixed_global

        saved_epoch = self._epoch
        counts = []
        for epoch in range(max(self.probe_epochs, 1)):
            self._epoch = epoch
            counts.append(len(self._pack_global_batches()))
        self._epoch = saved_epoch

        target = (min(counts) // world_size) * world_size
        if target <= 0:
            raise ValueError(
                "dataset does not contain one packed batch per DDP rank"
            )
        self._fixed_global = target
        if self.verbose and self._get_rank_info()[0] == 0:
            print(
                "  TokenBudgetBatchSampler: pinning every epoch to "
                f"{target} global batches ({target // world_size}/rank); "
                f"probe over {max(self.probe_epochs, 1)} epochs spanned "
                f"{min(counts)}-{max(counts)}",
                flush=True,
            )
        return target

    def _pack_global_batches(self):
        rng = random.Random(self.seed + self._epoch)
        indices = [
            index
            for index, size in enumerate(self.sizes)
            if not self.drop_oversized or size <= self.max_tokens
        ]
        indices.sort(key=lambda index: self.sizes[index])

        if self.shuffle:
            bucket_size = 80
            buckets = [
                indices[start : start + bucket_size]
                for start in range(0, len(indices), bucket_size)
            ]
            rng.shuffle(buckets)
            for bucket in buckets:
                rng.shuffle(bucket)
            indices = [index for bucket in buckets for index in bucket]

        batches = []
        current_batch = []
        current_tokens = 0
        for index in indices:
            event_tokens = self.sizes[index]
            if current_batch and current_tokens + event_tokens > self.max_tokens:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(index)
            current_tokens += event_tokens
        if current_batch and (not self.drop_last or not batches):
            batches.append(current_batch)

        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def _build_batches(self):
        rank, world_size = self._get_rank_info()
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank {rank} is incompatible with world_size={world_size}")
        batches = self._pack_global_batches()
        n_keep = (len(batches) // world_size) * world_size
        if self.stable_epoch_length and self.shuffle:
            target = self._stable_global_target(world_size)
            if len(batches) < target:
                raise RuntimeError(
                    f"epoch {self._epoch} packed into {len(batches)} batches, "
                    f"below the stable target {target}; increase probe_epochs"
                )
            n_keep = target
        batches = batches[:n_keep]
        return batches[rank::world_size]

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        self._cached_batches = self._build_batches()

    def __iter__(self):
        if self._cached_batches is None:
            self._cached_batches = self._build_batches()
        yield from self._cached_batches

    def __len__(self):
        if self._cached_batches is None:
            self._cached_batches = self._build_batches()
        return len(self._cached_batches)


class FixedEventBatchSampler(Sampler):
    """Distributed fixed-event batching with one shared deterministic order."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        shuffle: bool,
        drop_last: bool,
        seed: int = 42,
    ):
        self.num_events = len(dataset)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._epoch = 0
        self._cached_batches = None
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def _build_batches(self):
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size < 1 or rank < 0 or rank >= world_size:
            raise ValueError(
                f"rank {rank} is incompatible with world_size={world_size}"
            )
        indices = list(range(self.num_events))
        if self.shuffle:
            random.Random(self.seed + self._epoch).shuffle(indices)

        # Match DistributedSampler(drop_last=True) followed by local batching:
        # each rank gets an equal event shard, then drops its local partial batch.
        events_per_rank = self.num_events // world_size
        if not self.drop_last and self.num_events % world_size:
            events_per_rank += 1
            needed = events_per_rank * world_size - len(indices)
            repeats = (needed + len(indices) - 1) // len(indices)
            indices.extend((indices * repeats)[:needed])
        else:
            indices = indices[: events_per_rank * world_size]
        local = indices[rank : events_per_rank * world_size : world_size]
        batches = [
            local[start : start + self.batch_size]
            for start in range(0, len(local), self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        return batches

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        self._cached_batches = self._build_batches()

    def __iter__(self):
        if self._cached_batches is None:
            self._cached_batches = self._build_batches()
        yield from self._cached_batches

    def __len__(self):
        if self._cached_batches is None:
            self._cached_batches = self._build_batches()
        return len(self._cached_batches)
