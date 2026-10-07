import sys
from types import SimpleNamespace

import pytest
import torch

from shared_training.collation import collate_shared_events


def test_shared_collator_preserves_circe_particle_metadata():
    events = [
        {
            "features": torch.ones(2, 3),
            "positions": torch.ones(2, 3),
            "mc_index": torch.tensor([11, 11]),
            "is_secondary": torch.tensor([False, False]),
            "n_hits": 2,
            "particle_info": {11: {"pt": 1.5, "displacement": 25.0}},
        },
        {
            "features": torch.zeros(1, 3),
            "positions": torch.zeros(1, 3),
            "mc_index": torch.tensor([22]),
            "is_secondary": torch.tensor([False]),
            "n_hits": 1,
            "particle_info": {22: {"pt": 2.5, "displacement": 50.0}},
        },
    ]

    batch = collate_shared_events(events)

    assert batch["features"].shape == (3, 3)
    assert batch["positions"].shape == (3, 3)
    assert batch["seq_lens"] == [2, 1]
    assert batch["particle_info"] == [
        {11: {"pt": 1.5, "displacement": 25.0}},
        {22: {"pt": 2.5, "displacement": 50.0}},
    ]


def test_shared_collator_rejects_missing_circe_particle_metadata():
    event = {
        "features": torch.ones(1, 3),
        "positions": torch.ones(1, 3),
        "mc_index": torch.tensor([11]),
        "is_secondary": torch.tensor([False]),
        "n_hits": 1,
    }

    with pytest.raises(KeyError, match="particle_info"):
        collate_shared_events([event])


def test_shared_collator_builds_gatr_graph_batch_and_event_column(monkeypatch):
    class FakeGraph:
        def __init__(self, nodes):
            self.nodes = nodes

    class FakeBatch:
        def __init__(self, graphs):
            self.graphs = graphs

        def batch_num_nodes(self):
            return torch.tensor([graph.nodes for graph in self.graphs])

    monkeypatch.setitem(
        sys.modules,
        "dgl",
        SimpleNamespace(batch=lambda graphs: FakeBatch(graphs)),
    )
    first_graph = FakeGraph(2)
    second_graph = FakeGraph(1)
    first_truth = torch.tensor([[1.0, 11.0], [2.0, 12.0]])
    second_truth = torch.tensor([[3.0, 22.0]])

    graph, truth = collate_shared_events(
        [(first_graph, first_truth), (second_graph, second_truth)]
    )

    assert graph.batch_num_nodes().tolist() == [2, 1]
    assert truth.tolist() == [
        [1.0, 11.0, 0.0],
        [2.0, 12.0, 0.0],
        [3.0, 22.0, 1.0],
    ]
