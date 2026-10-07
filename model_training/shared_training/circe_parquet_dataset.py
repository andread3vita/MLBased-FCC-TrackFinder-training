"""CIRCE view of the shared event-wise Parquet schema also consumed by GATr.

The file stores raw detector measurements.  This loader constructs CIRCE's
full drift-circle representation at read time; GATr independently constructs
its midpoint/left-right representation from the very same rows.
"""

from __future__ import annotations

import glob
from functools import lru_cache
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from shared_training.collation import collate_shared_events

DRIFT_DIR_OFFSET = 10


def _normalize_time(values):
    return (
        torch.sign(values) * torch.log1p(values.abs()) / 5.0
    ).clamp_(-2.0, 2.5)


REQUIRED_COLUMNS = (
    "event_number", "file_number", "n_hit", "n_part",
    "hit_x_true", "hit_y_true", "hit_z_true", "hit_type",
    "hit_time", "hit_particle_index", "hit_x", "hit_y", "hit_z",
    "leftPosition_x", "leftPosition_y", "leftPosition_z",
    "rightPosition_x", "rightPosition_y", "rightPosition_z",
    "produced_by_secondary", "overlay", "drift_distance", "wire_azimuthal_angle",
    "wire_stereo_angle", "shared_schema_version",
    "part_p", "part_p_t", "part_theta", "part_phi", "part_m", "part_pid",
    "part_id", "gen_status", "part_parent",
    "part_vertex_x", "part_vertex_y", "part_vertex_z",
)


def expand_parquet_inputs(patterns):
    """Expand shell-independent file/glob arguments deterministically."""
    paths = []
    for pattern in patterns:
        matches = glob.glob(str(pattern))
        if matches:
            paths.extend(matches)
        elif Path(pattern).is_file():
            paths.append(str(pattern))
    unique = sorted(dict.fromkeys(str(Path(path).resolve()) for path in paths))
    if not unique:
        raise FileNotFoundError(f"No Parquet files matched: {patterns}")
    return unique


@lru_cache(maxsize=16)
def _read_row_group(path, row_group):
    table = pq.ParquetFile(path).read_row_group(row_group, columns=list(REQUIRED_COLUMNS))
    return ak.from_arrow(table)


class SharedParquetEventDataset(Dataset):
    """Common lazy index over the canonical event-wise Parquet rows."""

    def __init__(self, files, max_hits_per_event=None):
        self.files = expand_parquet_inputs(files)
        self.max_hits = max_hits_per_event
        self._index = []
        for path in self.files:
            parquet = pq.ParquetFile(path)
            missing = sorted(set(REQUIRED_COLUMNS) - set(parquet.schema_arrow.names))
            if missing:
                raise ValueError(f"{path} is not a shared-schema file; missing {missing}")
            for row_group in range(parquet.num_row_groups):
                sizes = parquet.read_row_group(row_group, columns=["n_hit"])[
                    "n_hit"
                ].to_pylist()
                for row, size in enumerate(sizes):
                    effective = min(int(size), max_hits_per_event) if max_hits_per_event else int(size)
                    self._index.append((path, row_group, row, effective))
        if not self._index:
            raise ValueError("The shared Parquet inputs contain no events")
        sizes = sorted(item[3] for item in self._index)
        print(
            f"SharedParquetEventDataset: {len(self._index)} events from "
            f"{len(self.files)} files; hits/event min={sizes[0]}, "
            f"median={sizes[len(sizes)//2]}, max={sizes[-1]}", flush=True,
        )

    def __len__(self):
        return len(self._index)

    def event(self, index):
        path, row_group, row, _ = self._index[index]
        return _read_row_group(path, row_group)[row]

    def event_key(self, index):
        path, row_group, row, _ = self._index[index]
        return path, row_group, row

    @staticmethod
    def _array(event, name, dtype=np.float32):
        return np.asarray(ak.to_numpy(event[name]), dtype=dtype)


class SharedIDEAParquetDataset(SharedParquetEventDataset):
    """CIRCE feature view of the common indexed Parquet events."""

    def __init__(self, files, max_hits_per_event=None, with_time=False,
                 with_drift_dir=False):
        super().__init__(files, max_hits_per_event=max_hits_per_event)
        self.with_time = bool(with_time)
        self.with_drift_dir = bool(with_drift_dir)

    def __getitem__(self, index):
        event = self.event(index)
        hit_type = self._array(event, "hit_type", np.int64)
        # Canonical schema: 1=planar (VTX/Si wrapper), 0=drift chamber.
        vtx_idx = np.flatnonzero(hit_type == 1)
        dc_idx = np.flatnonzero(hit_type == 0)
        if self.max_hits and len(vtx_idx) + len(dc_idx) > self.max_hits:
            keep_dc = max(self.max_hits - len(vtx_idx), 1)
            if keep_dc < len(dc_idx):
                rng = np.random.default_rng(index)
                dc_idx = np.sort(rng.choice(dc_idx, keep_dc, replace=False))
        order = np.concatenate((vtx_idx, dc_idx))
        n_vtx, n_dc = len(vtx_idx), len(dc_idx)
        n_cols = 10 + int(self.with_time) + 3 * int(self.with_drift_dir)
        features = torch.zeros((len(order), n_cols), dtype=torch.float32)

        measured = np.column_stack([
            self._array(event, name) for name in ("hit_x", "hit_y", "hit_z")
        ])
        true_pos = np.column_stack([
            self._array(event, name)
            for name in ("hit_x_true", "hit_y_true", "hit_z_true")
        ])
        features[:n_vtx, :3] = torch.from_numpy(measured[vtx_idx])
        features[n_vtx:, :3] = torch.from_numpy(true_pos[dc_idx])
        features[n_vtx:, 3] = 1.0
        if n_dc:
            wire = np.column_stack([
                self._array(event, name) for name in ("hit_x", "hit_y", "hit_z")
            ])
            drift = self._array(event, "drift_distance")
            azimuthal = self._array(event, "wire_azimuthal_angle")
            stereo = self._array(event, "wire_stereo_angle")
            dc_geometry = np.column_stack((wire, drift, azimuthal, stereo))
            features[n_vtx:, 4:10] = torch.from_numpy(dc_geometry[dc_idx])

        # Match GATr's display coordinates: measured planar position and the
        # midpoint of the two drift-circle candidates for chamber hits.
        left = np.column_stack([
            self._array(event, name)
            for name in ("leftPosition_x", "leftPosition_y", "leftPosition_z")
        ])
        right = np.column_stack([
            self._array(event, name)
            for name in ("rightPosition_x", "rightPosition_y", "rightPosition_z")
        ])
        display_positions = np.concatenate(
            (measured[vtx_idx], 0.5 * (left[dc_idx] + right[dc_idx])), axis=0
        )

        if self.with_time:
            time = torch.from_numpy(self._array(event, "hit_time")[order])
            features[:, 10] = _normalize_time(time)
        if self.with_drift_dir and n_dc:
            direction = torch.from_numpy(right[dc_idx] - left[dc_idx])
            direction = direction / direction.norm(dim=1, keepdim=True).clamp(min=1e-8)
            offset = DRIFT_DIR_OFFSET + int(self.with_time)
            features[n_vtx:, offset:offset + 3] = direction

        particle = self._array(event, "hit_particle_index", np.int64)[order]
        secondary = self._array(event, "produced_by_secondary", np.int64)[order]
        particle_ids = self._array(event, "part_id", np.int64)
        particle_pt = self._array(event, "part_p_t")
        particle_theta = self._array(event, "part_theta")
        particle_status = self._array(event, "gen_status", np.int64)
        particle_vertex_x = self._array(event, "part_vertex_x")
        particle_vertex_y = self._array(event, "part_vertex_y")
        particle_info = {
            int(particle_id): {
                "pt": float(pt),
                "theta": float(theta),
                "gen_status": int(status),
                "displacement": float(np.hypot(vertex_x, vertex_y)),
            }
            for particle_id, pt, theta, status, vertex_x, vertex_y in zip(
                particle_ids,
                particle_pt,
                particle_theta,
                particle_status,
                particle_vertex_x,
                particle_vertex_y,
            )
        }
        return {
            "features": features,
            "positions": torch.from_numpy(display_positions).float(),
            "mc_index": torch.from_numpy(particle).long(),
            "is_secondary": torch.from_numpy(secondary.astype(np.bool_)),
            "n_hits": len(order),
            "n_vtx": n_vtx,
            "n_dc": n_dc,
            "particle_info": particle_info,
        }


# Backward-compatible name for callers outside the matched-training launcher.
collate_idea_events = collate_shared_events
