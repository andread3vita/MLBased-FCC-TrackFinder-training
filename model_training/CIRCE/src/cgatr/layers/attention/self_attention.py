"""Self-attention layer for CGA Cl(4,1)."""

from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from src.cgatr.layers.attention.attention import GeometricAttention
from src.cgatr.layers.attention.config import SelfAttentionConfig
from src.cgatr.layers.attention.qkv import MultiQueryQKVModule
from src.cgatr.layers.dropout import GradeDropout
from src.cgatr.layers.linear import EquiLinear


class SelfAttention(nn.Module):
    """CGA geometric self-attention.

    Parameters
    ----------
    basis_q, basis_k : distance basis tensors (5, 5, 6)
    basis_pin : equivariant linear basis (9, 32, 32)
    config : SelfAttentionConfig
    """

    def __init__(self, basis_q, basis_k, basis_pin, config: SelfAttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.basis_pin = basis_pin

        # QKV computation (multi-query)
        self.qkv_module = MultiQueryQKVModule(self.basis_pin, config)

        # Output projection
        self.out_linear = EquiLinear(
            basis_pin=basis_pin,
            in_mv_channels=config.hidden_mv_channels * config.num_heads,
            out_mv_channels=config.out_mv_channels,
            in_s_channels=None
            if config.in_s_channels is None
            else config.hidden_s_channels * config.num_heads,
            out_s_channels=config.out_s_channels,
            initialization=config.output_init,
        )

        # No positional encoding by default (could add RoPE later)
        self.pos_encoding = nn.Identity()

        # Attention
        self.attention = GeometricAttention(basis_q, basis_k, config)

        # Dropout
        self.dropout: Optional[nn.Module]
        if config.dropout_prob is not None:
            self.dropout = GradeDropout(config.dropout_prob)
        else:
            self.dropout = None

    def forward(
        self,
        multivectors: torch.Tensor,
        additional_qk_features_mv: Optional[torch.Tensor] = None,
        scalars: Optional[torch.Tensor] = None,
        additional_qk_features_s: Optional[torch.Tensor] = None,
        attention_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        multivectors : torch.Tensor (..., num_items, channels_in, 32)
        scalars : torch.Tensor (..., num_items, s_channels)
        """
        q_mv, k_mv, v_mv, q_s, k_s, v_s = self.qkv_module(
            multivectors, scalars, additional_qk_features_mv, additional_qk_features_s
        )

        # Optional positional encoding on scalar q/k
        q_s = self.pos_encoding(q_s)
        k_s = self.pos_encoding(k_s)

        # Attention
        h_mv, h_s = self.attention(q_mv, k_mv, v_mv, q_s, k_s, v_s, attention_mask=attention_mask)

        # Rearrange heads
        h_mv = rearrange(
            h_mv, "... n_heads n_items hidden_channels x -> ... n_items (n_heads hidden_channels) x"
        )
        h_s = rearrange(
            h_s, "... n_heads n_items hidden_channels -> ... n_items (n_heads hidden_channels)"
        )

        # Output projection
        outputs_mv, outputs_s = self.out_linear(h_mv, scalars=h_s)

        # Dropout
        if self.dropout is not None:
            outputs_mv, outputs_s = self.dropout(outputs_mv, outputs_s)

        return outputs_mv, outputs_s
