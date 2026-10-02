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
    # Blade layout. Defaults describe Cl(4,1); the projective arm overrides them.
    grade1_idx: Optional[list] = None
    ip_idx: Optional[list] = None
    num_blades: int = 32
    # Metric weights for the attention inner product, aligned with `ip_idx`. The
    # geometric-algebra inner product is <x~ y>_0 = sum_i w_i x_i y_i, so an
    # unweighted dot product is only invariant when every w_i is +1. That holds
    # for the projective blades without e0 -- the shortcut GATr takes -- but not
    # in the conformal algebra, whose metric carries sixteen minus signs. None
    # reproduces the unweighted behaviour.
    ip_weights: Optional[list] = None
    # Trivector blade indices (homogeneous weight first) for GATr's
    # distance-aware attention features. Projective algebra only: the conformal
    # inner product already measures distance, the projective one provably
    # cannot. See de Haan et al. Prop. 3.
    pga_dist_idx: Optional[list] = None

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
