"""Utilities for optional truth-track pT weighting of training losses."""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch


DEFAULT_PT_TRACK_WEIGHT_BIN_EDGES = (0.4, 0.9, 5.0)
DEFAULT_PT_TRACK_WEIGHT_BIN_WEIGHTS = (1.5, 1.2, 0.75, 2.0)


def _parse_float_values(values, name: str) -> Tuple[float, ...]:
    if isinstance(values, str):
        fields = [field.strip() for field in values.split(",")]
        if not fields or any(not field for field in fields):
            raise ValueError(f"{name} must be a comma-separated list of numbers")
        try:
            parsed = tuple(float(field) for field in fields)
        except ValueError as exc:
            raise ValueError(
                f"{name} must be a comma-separated list of numbers"
            ) from exc
    else:
        parsed = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in parsed):
        raise ValueError(f"{name} values must be finite")
    return parsed


def parse_pt_track_weight_config(
    bin_edges: str | Iterable[float],
    bin_weights: str | Iterable[float],
) -> tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Parse and validate pT-bin edges and their per-track loss weights."""
    edges = _parse_float_values(bin_edges, "pT track-weight bin edges")
    weights = _parse_float_values(bin_weights, "pT track-weight bin weights")
    if any(edge <= 0.0 for edge in edges):
        raise ValueError("pT track-weight bin edges must be positive")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("pT track-weight bin edges must be strictly increasing")
    if len(weights) != len(edges) + 1:
        raise ValueError(
            "pT track weighting needs exactly one more weight than bin edges "
            f"(got {len(weights)} weights and {len(edges)} edges)"
        )
    if any(weight <= 0.0 for weight in weights):
        raise ValueError("pT track weights must be positive")
    return edges, weights


def pt_track_weights(
    pt: torch.Tensor,
    bin_edges: Iterable[float],
    bin_weights: Iterable[float],
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return one piecewise-constant loss weight per truth-track pT value."""
    edges, weights = parse_pt_track_weight_config(bin_edges, bin_weights)
    output_dtype = dtype if dtype is not None else pt.dtype
    finite_pt = torch.isfinite(pt)
    safe_pt = torch.where(finite_pt, pt.abs(), torch.zeros_like(pt))
    edge_tensor = torch.as_tensor(edges, dtype=safe_pt.dtype, device=pt.device)
    weight_tensor = torch.as_tensor(weights, dtype=output_dtype, device=pt.device)
    bin_index = torch.searchsorted(edge_tensor, safe_pt, right=True)
    result = weight_tensor[bin_index]
    # A malformed truth pT should not crash a complete batch or receive an
    # arbitrary extreme-bin weight. Its neutral fallback is one.
    return torch.where(finite_pt, result, torch.ones_like(result))


def weighted_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    """Mean values normally, or use a positive normalized weighted reduction."""
    if weights is None:
        return values.mean()
    if values.ndim != 1 or weights.ndim != 1 or values.shape != weights.shape:
        raise ValueError(
            "weighted_mean expects one-dimensional values and weights of equal shape"
        )
    typed_weights = weights.to(dtype=values.dtype, device=values.device)
    return torch.sum(values * typed_weights) / torch.sum(typed_weights)
