"""Geometric attention layer for CGA Cl(4,1)."""

from functools import partial

import torch
from torch import nn

from src.cgatr.layers.attention.config import SelfAttentionConfig
from src.cgatr.primitives.attention import geometric_attention, lin_square_normalizer


class GeometricAttention(nn.Module):
    """CGA geometric attention with distance-aware features.

    Parameters
    ----------
    basis_q, basis_k : torch.Tensor with shape (5, 5, 6)
        Distance basis tensors.
    config : SelfAttentionConfig
    """

    def __init__(self, basis_q, basis_k, config: SelfAttentionConfig) -> None:
        super().__init__()
        self.normalizer = partial(lin_square_normalizer, epsilon=config.normalizer_eps)
        self.log_weights = nn.Parameter(
            torch.zeros((config.num_heads, 1, config.hidden_mv_channels))
        )
        # Pass channel counts as Python ints so the inner kernel can
        # build its pad-to-multiple-of-8 channel count without reading
        # tensor shapes during ONNX tracing (which would emit
        # TracerWarnings).  The hidden mv/s channels are the same for Q
        # and V in multi-query attention (Q is reshaped to (heads,
        # items, hidden_mv) and K/V to (1, items, hidden_mv)); after
        # the heads dim is broadcast everyone has the same final
        # channel layout.
        h_mv = int(config.hidden_mv_channels)
        h_s = int(config.hidden_s_channels)
        self.geometric_attention = geometric_attention(
            basis_k, basis_q,
            num_mv_channels_qk=h_mv,
            num_s_channels_qk=h_s,
            num_mv_channels_v=h_mv,
            num_s_channels_v=h_s,
        )

    def forward(self, q_mv, k_mv, v_mv, q_s, k_s, v_s, attention_mask=None):
        weights = self.log_weights.exp()
        h_mv, h_s = self.geometric_attention(
            q_mv, k_mv, v_mv, q_s, k_s, v_s,
            normalizer=self.normalizer,
            weights=weights,
            attn_mask=attention_mask,
        )
        return h_mv, h_s
