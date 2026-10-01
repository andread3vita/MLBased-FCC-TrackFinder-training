"""Tests for the validation-only E(3)/SE(3) promotion rule."""

from src.eval.select_basis_gate import select_basis


def _rows(e3_def2, e3_fake, se3_def2, se3_fake):
    return {
        "e3_gate": {"def2": e3_def2, "ggtf10": e3_fake},
        "se3_gate": {"def2": se3_def2, "ggtf10": se3_fake},
    }


def test_only_def2_eligible_basis_wins():
    selected, _ = select_basis(_rows(70.0, 25.0, 66.0, 10.0))
    assert selected == "e3"


def test_lower_fake_rate_breaks_def2_eligible_tradeoff():
    selected, _ = select_basis(_rows(70.0, 30.0, 69.0, 20.0))
    assert selected == "se3"


def test_published_e3_wins_a_fake_rate_tie():
    selected, _ = select_basis(_rows(69.0, 20.8, 70.0, 20.0))
    assert selected == "e3"
