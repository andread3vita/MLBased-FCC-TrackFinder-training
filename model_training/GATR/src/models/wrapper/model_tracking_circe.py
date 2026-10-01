"""Network config selecting the CIRCE (conformal) model.

Usage:
    torchrun --nproc_per_node=4 -m src.train_lightning \
      --network-config src/models/wrapper/model_tracking_circe.py \
      ...
"""

import torch

from src.models.wrapper.model_tracking_cgatr import (
    GraphTransformerNetWrapper,
    get_model,
)

__all__ = ["GraphTransformerNetWrapper", "get_model"]
