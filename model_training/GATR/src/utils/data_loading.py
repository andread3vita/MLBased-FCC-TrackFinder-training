"""Small, dependency-free helpers for configuring PyTorch DataLoaders."""


def multiprocessing_loader_options(
    num_workers: int,
    prefetch_factor: int,
    *,
    persistent_workers: bool = False,
):
    """Return safe multiprocessing options for a DataLoader.

    DDP ranks have already initialized CUDA by the time their DataLoader
    iterators are created.  Linux's default ``fork`` context would therefore
    clone CUDA tensors and native CUDA/xFormers state into loader workers.
    ``spawn`` starts workers in clean interpreters while retaining parallel
    Parquet decoding and graph construction.
    """
    num_workers = int(num_workers)
    prefetch_factor = int(prefetch_factor)
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive")
    if num_workers == 0:
        return {}
    return {
        "multiprocessing_context": "spawn",
        "persistent_workers": bool(persistent_workers),
        "prefetch_factor": prefetch_factor,
    }
