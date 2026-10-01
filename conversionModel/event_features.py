#!/usr/bin/env python3
"""Build the seven GGTF inference features from one parquet event."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


MEASURED_COLUMNS = (
    "hit_x",
    "hit_y",
    "hit_z",
    "leftPosition_x",
    "leftPosition_y",
    "leftPosition_z",
    "rightPosition_x",
    "rightPosition_y",
    "rightPosition_z",
    "hit_type",
)


def load_real_event_features(parquet: Path, event_index: int) -> torch.Tensor:
    """Return ``[x, y, z, hit_type, vx, vy, vz]`` for one real event.

    This is the inference-only equivalent of ``create_graph_tracking_global``
    in the training data pipeline.  It intentionally reads no truth labels.
    Vertex hits use their measured position and a zero vector.  Drift-chamber
    hits use the midpoint of the left/right candidates and half their
    displacement.  Vertex hits precede drift-chamber hits, as in training.
    """
    parquet = Path(parquet)
    if not parquet.is_file():
        raise FileNotFoundError(parquet)
    if event_index < 0:
        raise ValueError("event_index must be non-negative")

    parquet_file = pq.ParquetFile(parquet)
    available = set(parquet_file.schema_arrow.names)
    missing = sorted(set(MEASURED_COLUMNS) - available)
    if missing:
        raise ValueError(f"Parquet file is missing measured columns: {missing}")
    if event_index >= parquet_file.metadata.num_rows:
        raise IndexError(
            f"Event index {event_index} is outside a file with "
            f"{parquet_file.metadata.num_rows} events"
        )

    table = parquet_file.read(columns=list(MEASURED_COLUMNS))
    columns = {
        name: np.asarray(table[name][event_index].as_py())
        for name in MEASURED_COLUMNS
    }
    lengths = {name: values.shape[0] for name, values in columns.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Measured hit columns have different lengths: {lengths}")

    measured = np.column_stack([columns[name] for name in MEASURED_COLUMNS])
    finite = np.isfinite(measured).all(axis=1)
    measured = measured[finite]
    if measured.shape[0] == 0:
        raise ValueError(f"Event {event_index} contains no finite measured hits")

    hit_type = measured[:, 9]
    if not np.all(np.isin(hit_type, (0, 1))):
        invalid = np.unique(hit_type[~np.isin(hit_type, (0, 1))]).tolist()
        raise ValueError(f"Event contains unsupported hit_type values: {invalid}")

    vertex = hit_type == 1
    drift = hit_type == 0
    vertex_position = measured[vertex, 0:3]
    vertex_vector = np.zeros_like(vertex_position)
    left = measured[drift, 3:6]
    right = measured[drift, 6:9]
    drift_position = 0.5 * (left + right)
    drift_vector = 0.5 * (right - left)

    position = np.concatenate((vertex_position, drift_position), axis=0)
    ordered_type = np.concatenate((hit_type[vertex], hit_type[drift]), axis=0)
    vector = np.concatenate((vertex_vector, drift_vector), axis=0)
    features = np.concatenate((position, ordered_type[:, None], vector), axis=1)
    return torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
