"""MLP configuration for CGA — identical to PGA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional


@dataclass
class MLPConfig:
    mv_channels: Optional[List[int]] = None
    s_channels: Optional[List[int]] = None
    activation: str = "gelu"
    dropout_prob: Optional[float] = None

    def __post_init__(self):
        if isinstance(self.dropout_prob, str) and self.dropout_prob.lower() in ["null", "none"]:
            self.dropout_prob = None

    @classmethod
    def cast(cls, config: Any) -> MLPConfig:
        if isinstance(config, MLPConfig):
            return config
        if isinstance(config, Mapping):
            return cls(**config)
        raise ValueError(f"Cannot cast {config} to {cls}")
