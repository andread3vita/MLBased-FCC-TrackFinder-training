import polars as pl
import torch

from src.dataset.parquet_dataset import ggtf_track_separation_weights
from src.model import object_condensation_loss


def _toy_loss_inputs():
    coords = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.1, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ])
    beta = torch.tensor([0.9, 0.2, 0.8, 0.1, 0.05])
    mc_index = torch.tensor([0, 0, 1, 1, -1])
    batch = torch.zeros(5, dtype=torch.long)
    return coords, beta, mc_index, batch


def test_default_oc_mode_is_unchanged_paper_hinge():
    args = _toy_loss_inputs()
    default = object_condensation_loss(*args, noise_index=-1)
    explicit = object_condensation_loss(
        *args, noise_index=-1, oc_mode="paper_hinge"
    )
    torch.testing.assert_close(default, explicit, rtol=0, atol=0)


def test_ggtf_mode_changes_charge_and_repulsion_but_not_attraction_formula():
    args = _toy_loss_inputs()
    _, paper = object_condensation_loss(
        *args, noise_index=-1, return_components=True
    )
    _, ggtf = object_condensation_loss(
        *args, noise_index=-1, oc_mode="ggtf", return_components=True
    )
    assert ggtf["L_V_rep"] > paper["L_V_rep"]
    assert ggtf["L_V_att"] < paper["L_V_att"]  # /1.01 lowers q weights


def test_ggtf_track_weights_reproduce_unwrapped_truth_eta_phi_rule():
    vtx = pl.DataFrame({
        "mc_index": [0, 1, 2],
        "produced_by_secondary": [False, False, False],
    })
    dc = pl.DataFrame({
        "mc_index": [0, 1, 2],
        "produced_by_secondary": [False, False, False],
    })
    particles = pl.DataFrame({
        "mc_index": [0, 1, 2],
        "theta": [torch.pi / 2] * 3,
        "phi": [0.0, 0.1, 0.5],
    })
    weights = ggtf_track_separation_weights(
        dc, vtx, particles, min_target_hits=2
    )
    expected = torch.tensor([
        1.0 / 0.101, 1.0 / 0.101, 1.0 / 0.401,
        1.0 / 0.101, 1.0 / 0.101, 1.0 / 0.401,
    ])
    torch.testing.assert_close(weights, expected, rtol=2e-5, atol=1e-5)
