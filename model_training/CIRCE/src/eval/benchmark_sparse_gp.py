"""Numerical, memory, and throughput gate for the optional sparse CGA GP."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from src.cgatr.primitives.bilinear import _GeometricProductFn
from src.cgatr.primitives.bilinear_triton import (
    cayley_sparse_tables,
    sparse_geometric_product,
)


def _timed_ms(fn, warmup: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), min(samples)


def _relative(got: torch.Tensor, want: torch.Tensor) -> float:
    return float(
        (got - want).abs().max() / want.abs().max().clamp_min(1.0e-12)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(42)
    device = torch.device("cuda")
    gp = torch.load(
        "cga_utils/cga_geometric_product.pt", weights_only=False
    ).to_dense().float().to(device)
    tables = tuple(table.to(device) for table in cayley_sparse_tables(gp))
    shape = (args.rows, args.channels, 32)
    x = torch.randn(shape, device=device)
    y = torch.randn(shape, device=device)
    grad = torch.randn(shape, device=device)

    def reference_values():
        xr = x.detach().requires_grad_(True)
        yr = y.detach().requires_grad_(True)
        out = _GeometricProductFn.apply(gp, xr, yr)
        gx, gy = torch.autograd.grad(out, (xr, yr), grad)
        return out, gx, gy

    def sparse_values():
        xs = x.detach().requires_grad_(True)
        ys = y.detach().requires_grad_(True)
        out = sparse_geometric_product(*tables, xs, ys)
        gx, gy = torch.autograd.grad(out, (xs, ys), grad)
        return out, gx, gy

    reference = reference_values()
    sparse = sparse_values()
    numerical = {
        "output_max_abs": float((sparse[0] - reference[0]).abs().max()),
        "output_relative": _relative(sparse[0], reference[0]),
        "grad_x_max_abs": float((sparse[1] - reference[1]).abs().max()),
        "grad_x_relative": _relative(sparse[1], reference[1]),
        "grad_y_max_abs": float((sparse[2] - reference[2]).abs().max()),
        "grad_y_relative": _relative(sparse[2], reference[2]),
    }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    reference_values()
    torch.cuda.synchronize()
    reference_peak = torch.cuda.max_memory_allocated()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sparse_values()
    torch.cuda.synchronize()
    sparse_peak = torch.cuda.max_memory_allocated()

    reference_median, reference_best = _timed_ms(
        reference_values, args.warmup, args.repeats
    )
    sparse_median, sparse_best = _timed_ms(
        sparse_values, args.warmup, args.repeats
    )
    result = {
        "shape": shape,
        "reference_median_ms": reference_median,
        "sparse_median_ms": sparse_median,
        "median_speedup": reference_median / sparse_median,
        "reference_best_ms": reference_best,
        "sparse_best_ms": sparse_best,
        "best_speedup": reference_best / sparse_best,
        "reference_peak_mib": reference_peak / 2**20,
        "sparse_peak_mib": sparse_peak / 2**20,
        "peak_memory_ratio": sparse_peak / max(reference_peak, 1),
        "numerical": numerical,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
