"""Training export must respect ranks and preserve usable resume checkpoints."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.cpu_export import CPUExportModelCheckpoint


def trainer_for(directory, *, rank_zero=True):
    def save(path, weights_only):
        Path(path).write_bytes(b"resume checkpoint")
    return SimpleNamespace(is_global_zero=rank_zero, sanity_checking=False,
                           global_step=12, loggers=[], save_checkpoint=save)


def mixed_features():
    features = torch.zeros(12, 10)
    features[6:, 3] = 1
    return features


def test_export_matches_saved_checkpoint_and_owns_example(tmp_path, monkeypatch):
    calls = []
    def export(checkpoint, features, library, output):
        assert Path(checkpoint).read_bytes() == b"resume checkpoint"
        assert features.shape == (8, 10)
        assert (features[:, 3] == 0).sum() == 4
        assert (features[:, 3] == 1).sum() == 4
        calls.append((checkpoint, library, output))
        Path(output).write_bytes(b"CPU graph")
        return "ema"
    monkeypatch.setattr("src.cpu_export.export_cpu_checkpoint", export)
    callback = CPUExportModelCheckpoint(dirpath=tmp_path, cpu_export_library="operator.so")
    trainer = trainer_for(tmp_path)
    features = mixed_features()
    callback.on_validation_batch_end(trainer, None, None, {"features": features}, 0)
    features.zero_()  # exporter must retain its own example after batch reuse
    path = str(tmp_path/"validation.ckpt")
    callback._save_checkpoint(trainer, path)
    assert calls == [(path, "operator.so", tmp_path/"validation.cpu.pt")]
    assert (tmp_path/"validation.cpu.pt").read_bytes() == b"CPU graph"


def test_only_rank_zero_exports(tmp_path, monkeypatch):
    def unexpected(*args):
        raise AssertionError("A nonzero rank attempted CPU export")
    monkeypatch.setattr("src.cpu_export.export_cpu_checkpoint", unexpected)
    callback = CPUExportModelCheckpoint(dirpath=tmp_path, cpu_export_library="operator.so")
    trainer = trainer_for(tmp_path, rank_zero=False)
    callback.on_validation_batch_end(trainer, None, None, {"features": mixed_features()}, 0)
    assert callback._cpu_example is None
    callback._save_checkpoint(trainer, str(tmp_path/"rank.ckpt"))


def test_export_failure_preserves_resume_checkpoint(tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("operator ABI mismatch")
    monkeypatch.setattr("src.cpu_export.export_cpu_checkpoint", fail)
    callback = CPUExportModelCheckpoint(dirpath=tmp_path, cpu_export_library="operator.so")
    trainer = trainer_for(tmp_path)
    callback.on_validation_batch_end(trainer, None, None, {"features": mixed_features()}, 0)
    path = str(tmp_path/"validation.ckpt")
    with pytest.warns(UserWarning, match="ABI mismatch"):
        callback._save_checkpoint(trainer, path)
    assert Path(path).read_bytes() == b"resume checkpoint"
    assert not (tmp_path/"validation.cpu.pt").exists()
