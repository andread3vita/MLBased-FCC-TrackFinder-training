"""GATr feature view of the common CIRCE-indexed event dataset."""

from __future__ import annotations

import numpy as np

from shared_training.circe_parquet_dataset import SharedParquetEventDataset
from src.dataset.functions_graph_tracking import create_graph_tracking_global
from src.utils.detector_features import DEFAULT_LAYERS_PER_SUPERLAYER


HIT_FEATURE_COLUMNS = (
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

PARTICLE_FEATURE_COLUMNS = (
    "part_theta",
    "part_phi",
    "part_m",
    "part_pid",
    "part_id",
    "part_p",
    "part_p_t",
    "gen_status",
    "part_parent",
    "part_vertex_x",
    "part_vertex_y",
    "part_vertex_z",
)


def _scalar(event, name):
    return int(np.asarray(event[name]))


class SharedGATrParquetDataset(SharedParquetEventDataset):
    """Build GATr graphs from the same indexed rows used by CIRCE.

    Only the event-to-feature transformation differs. Event selection, order,
    token sizes, batches, worker dispatch and DDP division are all controlled
    by the common CIRCE sampler.
    """

    def __init__(
        self,
        files,
        *,
        layers_per_superlayer=DEFAULT_LAYERS_PER_SUPERLAYER,
    ):
        super().__init__(files, max_hits_per_event=None)
        self.layers_per_superlayer = layers_per_superlayer

    def __getitem__(self, index):
        event = self.event(index)
        declared_n_hits = _scalar(event, "n_hit")
        declared_n_particles = _scalar(event, "n_part")
        event_number = _scalar(event, "event_number")
        file_number = _scalar(event, "file_number")

        hits_features = np.stack(
            [self._array(event, name) for name in HIT_FEATURE_COLUMNS], axis=0
        ).astype(np.float32, copy=False)
        particle_features = np.stack(
            [self._array(event, name) for name in PARTICLE_FEATURE_COLUMNS],
            axis=0,
        ).astype(np.float32, copy=False)
        hit_labels = self._array(
            event, "hit_particle_index", np.int64
        )[None, :]
        # Match CIRCE: the declared counts control the common batch plan, while
        # feature construction consumes the arrays actually stored in the row.
        # Do not reject a row early merely because a declared count is stale.
        n_hits = hits_features.shape[1]
        n_particles = particle_features.shape[1]
        if n_hits != declared_n_hits or n_particles != declared_n_particles:
            print(
                "WARNING: preserving shared event whose declared counts differ "
                f"from stored arrays: index={index}, "
                f"n_hit={declared_n_hits}/{n_hits}, "
                f"n_part={declared_n_particles}/{n_particles}",
                flush=True,
            )

        width = max(n_hits, n_particles, 1)
        mask = np.zeros((6, width), dtype=np.float32)
        mask[0, :n_hits] = 1.0
        mask[1, :n_particles] = 1.0
        mask[2, :n_hits] = self._array(event, "produced_by_secondary")
        mask[3, :n_hits] = event_number
        mask[4, :n_hits] = file_number
        mask[5, :n_hits] = self._array(event, "overlay")

        output = {
            "hits_features": hits_features,
            "particle_features": particle_features,
            "hits_labels": hit_labels,
            "mask": mask,
        }
        graph_and_truth, graph_empty = create_graph_tracking_global(
            output,
            file_number,
            event_number,
            get_vtx=True,
            vector=True,
            overlay=False,
            layers_per_superlayer=self.layers_per_superlayer,
            keep_all_events=True,
        )
        if graph_empty:
            path, row_group, row = self.event_key(index)
            raise ValueError(
                "The common loader cannot build the planned GATr event "
                f"{path}:row_group={row_group}:row={row}; refusing to silently "
                "substitute or skip an event that CIRCE will train on"
            )
        return graph_and_truth
