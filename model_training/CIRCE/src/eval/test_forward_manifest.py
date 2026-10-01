"""Inference must not silently change a checkpoint's input representation."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.eval import forward_pass
from src.eval.forward_pass import (
    _checkpoint_fingerprint,
    _signal_policy_matches,
    _validate_checkpoint_config,
)


def _args(**overrides):
    values = dict(
        num_blocks=10,
        hidden_mv_channels=16,
        hidden_s_channels=64,
        embed_dim=4,
        normalize_mv_inputs=False,
        algebra="conformal",
        use_time=False,
        pga_hit_encoding="line",
        two_channel_dc=False,
        cga_hit_encoding="sphere_circle",
        physical_drift_geometry=True,
        separate_hit_metadata=False,
        separate_hit_type=False,
        layernorm_epsilon_mode="clamp",
        no_legacy_equivariance=True,
        equivariance_group="e3",
        invariant_output_head=True,
        fix_cga_null=True,
        fix_wire_dir=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _checkpoint():
    args = _args()
    saved = vars(args).copy()
    saved["legacy_equivariance"] = not saved.pop("no_legacy_equivariance")
    return {"hyper_parameters": saved}


def test_matching_checkpoint_configuration_is_accepted():
    _validate_checkpoint_config(_checkpoint(), _args())


def test_input_l2_mismatch_is_rejected():
    with pytest.raises(RuntimeError, match="normalize_mv_inputs"):
        _validate_checkpoint_config(
            _checkpoint(), _args(normalize_mv_inputs=True)
        )


def test_encoding_mismatch_is_rejected():
    with pytest.raises(RuntimeError, match="cga_hit_encoding"):
        _validate_checkpoint_config(
            _checkpoint(), _args(cga_hit_encoding="circle")
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("separate_hit_type", True),
        ("layernorm_epsilon_mode", "add"),
    ],
)
def test_ablation_mismatch_is_rejected(field, value):
    with pytest.raises(RuntimeError, match=field):
        _validate_checkpoint_config(
            _checkpoint(), _args(**{field: value})
        )


def test_checkpoint_fingerprint_tracks_content_not_mutable_path(tmp_path):
    checkpoint = tmp_path / "cgatr_best.ckpt"
    checkpoint.write_bytes(b"epoch 18")
    first = _checkpoint_fingerprint(str(checkpoint))
    checkpoint.write_bytes(b"epoch 19")
    second = _checkpoint_fingerprint(str(checkpoint))
    assert first != second


def test_legacy_manifest_defaults_to_particle_zero_excluded(monkeypatch):
    monkeypatch.setattr(forward_pass, "MIN_SIGNAL_MC", 1)
    assert _signal_policy_matches({})
    assert not _signal_policy_matches({"min_signal_mc": 0})


def test_particle_zero_policy_is_part_of_cache_identity(monkeypatch):
    monkeypatch.setattr(forward_pass, "MIN_SIGNAL_MC", 0)
    assert _signal_policy_matches({"min_signal_mc": 0})
    assert not _signal_policy_matches({})


def test_forward_cache_has_unique_hit_order_within_event():
    frame = forward_pass.cache_to_dataframe([{
        "event_id": 7,
        "seed": 181,
        "sig_coords": np.zeros((3, 4), dtype=np.float32),
        "sig_beta": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "sig_mc": np.array([4, 4, 5], dtype=np.int64),
        "n_hits_total_map": {4: 2, 5: 1},
    }], embed_dim=4)

    assert frame["hit_order"].to_list() == [0, 1, 2]
    assert frame.select(["seed", "event_id", "hit_order"]).unique().height == 3
