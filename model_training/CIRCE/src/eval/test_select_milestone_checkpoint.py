import json

import pytest

from src.eval.select_milestone_checkpoint import select_milestone


def _summary(root, arm, tbeta, td, def2, fake, eff=0.9):
    out = root / arm / f"tb{tbeta}_td{td}_n0"
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps({
        "tbeta": tbeta,
        "td": td,
        "n_events": 100,
        "n_tracks_total": 1000,
        "candidates_per_event": 30.0,
        "no_cuts": {"efficiency": eff, "def2": def2 / 100.0},
        "ggtf_fake_rate_gt10": fake / 100.0,
        "ggtf_n_matched_gt10": 100,
        "ggtf_n_fake_clone_gt10": 4,
        "ggtf_n_fake_spurious_gt10": 1,
    }))


def test_selection_pools_all_grid_points_before_applying_tolerance(tmp_path):
    sweep = tmp_path / "sweep"
    run = tmp_path / "run"
    run.mkdir()
    (run / "cgatr_epoch07.ckpt").write_bytes(b"epoch 8")
    (run / "cgatr_epoch15.ckpt").write_bytes(b"epoch 16")

    _summary(sweep, "corrected_e3_epoch08", 0.1, 0.1, 80.0, 3.0)
    _summary(sweep, "corrected_e3_epoch08", 0.2, 0.1, 78.5, 1.0)
    _summary(sweep, "corrected_e3_epoch16", 0.1, 0.1, 79.0, 1.5)
    # This wins an epoch-16-only selection, but is outside two points of the
    # global best and therefore must not win the campaign.
    _summary(sweep, "corrected_e3_epoch16", 0.2, 0.1, 77.0, 0.5)

    selected = select_milestone(
        str(sweep),
        ["corrected_e3_epoch08", "corrected_e3_epoch16"],
        str(run),
    )

    assert selected["selected_arm"] == "corrected_e3_epoch08"
    assert selected["epoch"] == 8
    assert selected["operating_point"] == {"tbeta": 0.2, "td": 0.1}
    assert selected["validation_metrics_percent"]["ggtf10"] == 1.0
    assert len(selected["checkpoint_sha256"]) == 64


def test_arm_name_must_encode_epoch(tmp_path):
    with pytest.raises(ValueError, match="_epochNN"):
        select_milestone(str(tmp_path), ["final"], str(tmp_path))
