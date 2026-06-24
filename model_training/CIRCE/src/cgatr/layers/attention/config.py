"""Attention configuration for CGA Cl(4,1) — identical structure to PGA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass
class SelfAttentionConfig:
    """Configuration for CGA self-attention.

    Parameters
    ----------
    multi_query : bool
        Whether to use multi-query attention.
    in_mv_channels : int
    out_mv_channels : int
    in_s_channels : int
    out_s_channels : int
    num_heads : int
    additional_qk_mv_channels : int
    additional_qk_s_channels : int
    normalizer_eps : float
    pos_encoding : bool
    pos_enc_base : int
    output_init : str
    checkpoint : bool
    increase_hidden_channels : int
    dropout_prob : float or None
    """

    multi_query: bool = True
    in_mv_channels: Optional[int] = None
    out_mv_channels: Optional[int] = None
    in_s_channels: Optional[int] = None
    out_s_channels: Optional[int] = None
    num_heads: int = 8
    additional_qk_mv_channels: int = 0
    additional_qk_s_channels: int = 0
    normalizer_eps: Optional[float] = 1e-3
    pos_encoding: bool = False
    pos_enc_base: int = 4096
    output_init: str = "default"
    checkpoint: bool = True
    increase_hidden_channels: int = 2
    dropout_prob: Optional[float] = None

    def __post_init__(self):
        if isinstance(self.dropout_prob, str) and self.dropout_prob.lower() in ["null", "none"]:
            self.dropout_prob = None

    @property
    def hidden_mv_channels(self) -> Optional[int]:
        if self.in_mv_channels is None:
            return None
        return max(self.increase_hidden_channels * self.in_mv_channels // self.num_heads, 1)

    @property
    def hidden_s_channels(self) -> Optional[int]:
        if self.in_s_channels is None:
            return None
        hidden_s_channels = max(
            self.increase_hidden_channels * self.in_s_channels // self.num_heads, 4
        )
        if self.pos_encoding:
            hidden_s_channels = (hidden_s_channels + 1) // 2 * 2
            hidden_s_channels = max(hidden_s_channels, 8)
        return hidden_s_channels

    @classmethod
    def cast(cls, config: Any) -> SelfAttentionConfig:
        if isinstance(config, SelfAttentionConfig):
            return config
        if isinstance(config, Mapping):
            return cls(**config)
        raise ValueError(f"Cannot cast {config} to {cls}")
