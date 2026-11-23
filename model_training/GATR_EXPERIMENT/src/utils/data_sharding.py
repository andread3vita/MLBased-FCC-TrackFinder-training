def shard_event_indices(indices, shard_rank: int, num_shards: int):
    """Partition ordered event indices into deterministic, disjoint shards."""
    shard_rank = int(shard_rank)
    num_shards = int(num_shards)
    if num_shards < 1:
        raise ValueError("num_shards must be at least one")
    if shard_rank < 0 or shard_rank >= num_shards:
        raise ValueError(
            f"shard_rank={shard_rank} must be in [0, {num_shards})"
        )
    return indices[shard_rank::num_shards]


def shard_file_dict(file_dict, shard_rank: int, num_shards: int):
    """Partition every named file group into deterministic rank shards."""
    shard_rank = int(shard_rank)
    num_shards = int(num_shards)
    if num_shards < 1:
        raise ValueError("num_shards must be at least one")
    if shard_rank < 0 or shard_rank >= num_shards:
        raise ValueError(
            f"shard_rank={shard_rank} must be in [0, {num_shards})"
        )

    sharded = {}
    for name, files in file_dict.items():
        rank_files = list(files)[shard_rank::num_shards]
        if not rank_files:
            raise ValueError(
                f"File group {name!r} has {len(files)} file(s), fewer than "
                f"the {num_shards} distributed ranks"
            )
        sharded[name] = rank_files
    return sharded
