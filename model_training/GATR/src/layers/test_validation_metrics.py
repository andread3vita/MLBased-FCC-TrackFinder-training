import torch

from src.layers.validation_metrics import compute_batch_metrics_greedy


def test_strict50_is_one_for_two_well_separated_tracks():
    metrics = compute_batch_metrics_greedy(
        coords=torch.tensor([
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
            [2.0, 0.0], [2.1, 0.0], [2.0, 0.1],
        ]),
        beta_logits=torch.tensor([[10.0], [-10.0], [-10.0],
                                  [9.0], [-10.0], [-10.0]]),
        mc_index=torch.tensor([1, 1, 1, 2, 2, 2]),
        is_secondary=torch.zeros(6, dtype=torch.bool),
        seq_lens=[6],
        tbeta=0.5,
        td=0.3,
    )

    assert metrics["match_rate"] == 1.0
    assert metrics["match_rate_strict50"] == 1.0


def test_strict50_requires_purity_strictly_above_three_quarters():
    metrics = compute_batch_metrics_greedy(
        coords=torch.tensor([
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
            [0.1, 0.1], [2.0, 0.0], [2.1, 0.0],
        ]),
        beta_logits=torch.tensor([[10.0], [-10.0], [-10.0],
                                  [-10.0], [9.0], [-10.0]]),
        mc_index=torch.tensor([1, 1, 1, 2, 2, 2]),
        is_secondary=torch.zeros(6, dtype=torch.bool),
        seq_lens=[6],
        tbeta=0.5,
        td=0.3,
    )

    # Track 1 has purity exactly 3/4 and therefore fails; track 2 passes.
    assert metrics["match_rate"] == 0.5
    assert metrics["match_rate_strict50"] == 0.5
