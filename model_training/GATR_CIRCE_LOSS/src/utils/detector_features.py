"""Detector-scalar layout and global-layer construction."""

from __future__ import annotations

from collections.abc import Iterable

import torch


NUM_SUPERLAYERS = 14
DEFAULT_LAYERS_PER_SUPERLAYER = (8,) * NUM_SUPERLAYERS

# These are the columns loaded from the Parquet files. ``layer`` and
# ``superLayer`` are construction inputs and are never passed separately to
# the model.
DETECTOR_SOURCE_FEATURE_NAMES = (
    "hit_time",
    "layer",
    "superLayer",
    "stereo",
    "cluster_count",
)

DETECTOR_FEATURE_NAMES = (
    "hit_time",
    "global_layer",
    "stereo",
    "cluster_count",
)


def validate_layers_per_superlayer(values: Iterable[int]) -> tuple[int, ...]:
    """Return a validated 14-entry detector layout."""
    layout = tuple(int(value) for value in values)
    if len(layout) != NUM_SUPERLAYERS:
        raise ValueError(
            "layers_per_superlayer must contain exactly "
            f"{NUM_SUPERLAYERS} entries; received {len(layout)}"
        )
    if any(value <= 0 for value in layout):
        raise ValueError(
            "layers_per_superlayer entries must all be positive; "
            f"received {layout}"
        )
    return layout


def build_detector_scalars(
    source_features: torch.Tensor,
    layers_per_superlayer: Iterable[int],
) -> torch.Tensor:
    """Replace local layer/superlayer columns by a cumulative global layer."""
    if source_features.ndim != 2:
        raise ValueError(
            "Detector source features must be a matrix; received shape "
            f"{tuple(source_features.shape)}"
        )
    if source_features.shape[1] == 0:
        return source_features
    if source_features.shape[1] != len(DETECTOR_SOURCE_FEATURE_NAMES):
        raise ValueError(
            f"Expected detector source columns {DETECTOR_SOURCE_FEATURE_NAMES}; "
            f"received a matrix with {source_features.shape[1]} columns"
        )

    layout = validate_layers_per_superlayer(layers_per_superlayer)
    layer = source_features[:, 1]
    superlayer = source_features[:, 2]
    rounded_layer = torch.round(layer)
    rounded_superlayer = torch.round(superlayer)
    if not torch.allclose(layer, rounded_layer, rtol=0.0, atol=1.0e-6):
        raise ValueError("Detector layer values must be integer-valued")
    if not torch.allclose(superlayer, rounded_superlayer, rtol=0.0, atol=1.0e-6):
        raise ValueError("Detector superLayer values must be integer-valued")

    layer_index = rounded_layer.to(dtype=torch.long)
    superlayer_index = rounded_superlayer.to(dtype=torch.long)
    if torch.any((superlayer_index < 0) | (superlayer_index >= NUM_SUPERLAYERS)):
        invalid = torch.unique(
            superlayer_index[
                (superlayer_index < 0) | (superlayer_index >= NUM_SUPERLAYERS)
            ]
        ).tolist()
        raise ValueError(
            f"superLayer indices must be in [0, {NUM_SUPERLAYERS - 1}]; "
            f"received {invalid}"
        )

    counts = torch.as_tensor(layout, device=source_features.device, dtype=torch.long)
    per_hit_counts = counts[superlayer_index]
    invalid_layer = (layer_index < 0) | (layer_index >= per_hit_counts)
    if torch.any(invalid_layer):
        examples = torch.stack(
            (superlayer_index[invalid_layer], layer_index[invalid_layer]), dim=1
        )[:5].tolist()
        raise ValueError(
            "layer indices must satisfy 0 <= layer < "
            "layers_per_superlayer[superLayer]; invalid "
            f"(superLayer, layer) examples: {examples}"
        )

    offsets = torch.cat((counts.new_zeros(1), torch.cumsum(counts[:-1], dim=0)))
    global_layer = (offsets[superlayer_index] + layer_index).to(
        dtype=source_features.dtype
    )
    return torch.stack(
        (
            source_features[:, 0],
            global_layer,
            source_features[:, 3],
            source_features[:, 4],
        ),
        dim=1,
    )
