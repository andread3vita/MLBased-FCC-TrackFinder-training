from src.dataset.parquet_ggtf_adapter import TokenBudgetEventSampler


def test_cached_global_batches_are_rebuilt_when_ddp_context_appears(monkeypatch):
    """Regression test for the full-run epoch-boundary NCCL timeout."""
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    sampler = TokenBudgetEventSampler(
        [9] * 8, max_tokens=10, shuffle=False, verbose=False
    )
    assert len(sampler) == 8

    # Lightning direct DDP sets these after constructing rank 0. The old
    # sampler retained the eight-batch global cache while rank 1 saw four.
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert len(sampler) == 4


def test_explicit_replicas_partition_batches_before_ddp_initializes(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    rank0 = TokenBudgetEventSampler(
        [9] * 8, max_tokens=10, shuffle=False, verbose=False,
        num_replicas=2, rank=0,
    )
    rank1 = TokenBudgetEventSampler(
        [9] * 8, max_tokens=10, shuffle=False, verbose=False,
        num_replicas=2, rank=1,
    )

    batches0 = list(rank0)
    batches1 = list(rank1)
    assert len(batches0) == len(batches1) == 4
    assert set(map(tuple, batches0)).isdisjoint(map(tuple, batches1))
    assert sorted(i for batch in batches0 + batches1 for i in batch) == list(range(8))
