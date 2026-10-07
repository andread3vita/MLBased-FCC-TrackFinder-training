"""One collator entry point for the matched CIRCE/GATr data path.

The event-to-feature transforms remain model-specific: CIRCE dataset items are
mapping-like tensor records, while GATr dataset items are ``(graph, truth)``
pairs.  Both shared trainers nevertheless call this same function, which
preserves the common event batch and dispatches only the final representation.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


def _collate_circe_events(events):
    required = {
        "features", "positions", "mc_index", "is_secondary", "n_hits",
        "particle_info",
    }
    missing = [sorted(required - set(event)) for event in events]
    if any(missing):
        raise KeyError(f"CIRCE shared events are missing required fields: {missing}")

    collated = {
        "features": torch.cat([event["features"] for event in events], dim=0),
        "positions": torch.cat([event["positions"] for event in events], dim=0),
        "mc_index": torch.cat([event["mc_index"] for event in events], dim=0),
        "is_secondary": torch.cat(
            [event["is_secondary"] for event in events], dim=0
        ),
        "seq_lens": [event["n_hits"] for event in events],
        # Validation needs this metadata for efficiency versus pT and
        # production displacement. Never silently discard it on the shared path.
        "particle_info": [event["particle_info"] for event in events],
    }
    if all("track_separation_weight" in event for event in events):
        collated["track_separation_weight"] = torch.cat(
            [event["track_separation_weight"] for event in events], dim=0
        )
    return collated


def _collate_gatr_events(events):
    # Import lazily so CIRCE workers do not need to initialize DGL merely to
    # collate their tensor representation.
    import dgl

    graphs = [event[0] for event in events]
    truth_rows = []
    for batch_id, event in enumerate(events):
        truth = event[1]
        event_column = truth.new_full((truth.shape[0], 1), batch_id)
        truth_rows.append(torch.cat((truth, event_column), dim=1))
    return dgl.batch(graphs), torch.cat(truth_rows, dim=0)


def collate_shared_events(batch):
    """Collate either model's canonical shared-schema event representation."""
    events = [event for event in batch if event is not None]
    if not events:
        return None

    first_is_mapping = isinstance(events[0], Mapping)
    if not all(isinstance(event, Mapping) == first_is_mapping for event in events):
        raise TypeError("Cannot collate a mixture of CIRCE and GATr event records")
    if first_is_mapping:
        return _collate_circe_events(events)
    if all(isinstance(event, (tuple, list)) and len(event) >= 2 for event in events):
        return _collate_gatr_events(events)
    raise TypeError(f"Unsupported shared event record: {type(events[0]).__name__}")
