"""Token-budget batching for GATr's streaming event dataset.

CIRCE packs events until their total hit count reaches ``max_tokens``.  GATr
keeps its existing lazy iterable dataset and file-fetching implementation, but
this wrapper applies the same budget to the graphs produced by that stream.
"""

from __future__ import annotations

import random
from functools import partial

import torch


class TokenBudgetIterableDataset(torch.utils.data.IterableDataset):
    """Group streamed GATr events into batches bounded by total graph nodes."""

    def __init__(self, dataset, max_tokens, *, drop_last, target_batches=None):
        super().__init__()
        self.dataset = dataset
        self.max_tokens = int(max_tokens)
        self.drop_last = bool(drop_last)
        self.target_batches = (
            None if target_batches is None else int(target_batches)
        )
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    @property
    def config(self):
        return self.dataset.config

    def __len__(self):
        if self.target_batches is None:
            raise TypeError("streaming token-budget dataset has no fixed length")
        return self.target_batches

    def __iter__(self):
        batch = []
        tokens = 0
        yielded = 0
        for event in self.dataset:
            event_tokens = int(event[0].num_nodes())
            if batch and tokens + event_tokens > self.max_tokens:
                yield batch
                yielded += 1
                if (
                    self.target_batches is not None
                    and yielded >= self.target_batches
                ):
                    return
                batch = []
                tokens = 0
            batch.append(event)
            tokens += event_tokens
        if batch and not self.drop_last:
            yield batch


def _apply_collator(events, collator):
    return collator(events)


def token_budget_collator(collator):
    """Return a spawn-picklable collator for pre-batched iterable samples."""
    return partial(_apply_collator, collator=collator)


def _packed_batch_count(sizes, max_tokens, epoch):
    """Count CIRCE-style size-sorted, bucket-shuffled packed batches."""
    rng = random.Random(42 + int(epoch))
    ordered = sorted(int(size) for size in sizes)
    bucket_size = 80
    buckets = [
        ordered[index:index + bucket_size]
        for index in range(0, len(ordered), bucket_size)
    ]
    rng.shuffle(buckets)
    for bucket in buckets:
        rng.shuffle(bucket)
    ordered = [size for bucket in buckets for size in bucket]

    batches = 0
    current_tokens = 0
    has_current = False
    for event_tokens in ordered:
        if has_current and current_tokens + event_tokens > max_tokens:
            batches += 1
            current_tokens = 0
            has_current = False
        current_tokens += event_tokens
        has_current = True

    # Match CIRCE's training sampler: drop the final partial packed batch, but
    # retain a singleton dataset as one usable batch.
    if has_current and batches == 0:
        batches = 1
    return batches


def read_event_sizes(files):
    """Read the canonical ``n_hit`` metadata used by CIRCE's batch planner."""
    import pyarrow.parquet as pq

    sizes = []
    for file_spec in files:
        path = str(file_spec).split(":", 1)[-1]
        parquet = pq.ParquetFile(path)
        if "n_hit" not in parquet.schema_arrow.names:
            raise ValueError(
                f"{path} has no n_hit column required by --max-tokens; "
                "use --max-tokens 0 for fixed event-count batching"
            )
        for row_group in range(parquet.num_row_groups):
            sizes.extend(
                int(value)
                for value in parquet.read_row_group(
                    row_group, columns=["n_hit"]
                )["n_hit"].to_pylist()
            )
    if not sizes:
        raise ValueError("cannot plan token-budget batches for an empty dataset")
    return sizes


def stable_steps_per_rank(files, max_tokens, world_size, probe_epochs=64):
    """Return CIRCE's stable packed-batch count divided across DDP ranks."""
    max_tokens = int(max_tokens)
    world_size = int(world_size)
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    sizes = read_event_sizes(files)
    counts = [
        _packed_batch_count(sizes, max_tokens, epoch)
        for epoch in range(max(int(probe_epochs), 1))
    ]
    global_batches = (min(counts) // world_size) * world_size
    if global_batches <= 0:
        raise ValueError(
            "token-budget dataset does not contain one batch per DDP rank"
        )
    return global_batches // world_size, len(sizes), min(counts), max(counts)
