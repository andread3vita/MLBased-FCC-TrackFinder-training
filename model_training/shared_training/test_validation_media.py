import sys
from types import ModuleType

import numpy as np

from shared_training import validation_media

# The lightweight test environment has Plotly but not Lightning. Supply the
# logger base needed to import the pure media-filter helper under test.
if "lightning.pytorch.loggers" not in sys.modules:
    lightning = ModuleType("lightning")
    pytorch = ModuleType("lightning.pytorch")
    loggers = ModuleType("lightning.pytorch.loggers")
    loggers.WandbLogger = object
    lightning.pytorch = pytorch
    pytorch.loggers = loggers
    sys.modules.setdefault("lightning", lightning)
    sys.modules.setdefault("lightning.pytorch", pytorch)
    sys.modules.setdefault("lightning.pytorch.loggers", loggers)
from shared_training.wandb_logger import log_wandb_media


def test_hit_media_is_saved_and_embedding_can_be_disabled(tmp_path, monkeypatch):
    class Figure:
        def to_html(self, **kwargs):
            return "<div>interactive plot</div>"

    monkeypatch.setattr(
        validation_media,
        "validation_scatter_figure",
        lambda *args, **kwargs: Figure(),
    )
    monkeypatch.setattr(
        validation_media,
        "greedy_cluster",
        lambda *args, **kwargs: np.array([0, 0, 1]),
    )
    event = {
        "positions": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]]),
        "coords": np.array([[0, 0], [0.1, 0], [1, 1]]),
        "beta": np.array([0.9, 0.8, 0.7]),
        "mc_particle_id": np.array([11, 11, 22]),
    }
    working_point = {"tbeta": 0.5, "td": 0.2, "min_hits": 1}

    media = validation_media.validation_event_media(
        event,
        working_point,
        rejected_seed_policy="attach-after-accept",
        output_dir=tmp_path,
        include_embedding=False,
    )

    assert set(media) == {
        "plots/validation_event_0/hits_by_mc_particle",
        "plots/validation_event_0/hits_by_reconstructed_particle",
    }
    assert (tmp_path / "validation_event_0_hits_by_mc_particle.html").is_file()
    assert (
        tmp_path / "validation_event_0_hits_by_reconstructed_particle.html"
    ).is_file()


def test_wandb_filter_accepts_only_shared_validation_hit_media():
    class Experiment:
        def __init__(self):
            self.logged = None

        def log(self, media):
            self.logged = media

    class Logger:
        def __init__(self):
            self.experiment = Experiment()

    logger = Logger()
    log_wandb_media(logger, {
        "plots/validation_event_0/hits_by_mc_particle": "hits-mc",
        "plots/validation_event_0/hits_by_reconstructed_particle": "hits-reco",
        "plots/validation_event_0/embedding_by_mc_particle": "embedding",
    })

    assert logger.experiment.logged == {
        "plots/validation_event_0/hits_by_mc_particle": "hits-mc",
        "plots/validation_event_0/hits_by_reconstructed_particle": "hits-reco",
    }
