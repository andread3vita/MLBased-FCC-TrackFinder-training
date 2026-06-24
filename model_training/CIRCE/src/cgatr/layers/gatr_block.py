"""CGA transformer block — IPNS two-channel architecture.

With two-channel input (sphere grade-1 + circle grade-2), channel 0 is always
grade-1 for all hit types, so cross-attention works natively without grade mixing.
The GP in the MLP naturally propagates circle geometry from channel 1.
"""

from dataclasses import replace
from typing import Optional, Tuple

import torch
from torch import nn

from src.cgatr.layers import SelfAttention, SelfAttentionConfig
from src.cgatr.layers.layer_norm import EquiLayerNorm
from src.cgatr.layers.mlp.config import MLPConfig
from src.cgatr.layers.mlp.mlp import GeoMLP


class CGATrBlock(nn.Module):
    """CGA equivariant transformer block.

    Architecture: norm → attention → + → norm → MLP → +

    No grade mixer needed — the two-channel IPNS embedding ensures
    channel 0 is always grade-1 for all hit types.
    """

    def __init__(
        self,
        gp,
        ip_weights,
        outer,
        basis_pin,
        basis_q,
        basis_k,
        mv_channels: int,
        s_channels: int,
        attention: SelfAttentionConfig,
        mlp: MLPConfig,
        dropout_prob: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.norm = EquiLayerNorm(ip_weights, hidden_s_channels=s_channels)
        self.norm2 = EquiLayerNorm(ip_weights, hidden_s_channels=s_channels)

        # Self-attention
        attention = replace(
            attention,
            in_mv_channels=mv_channels,
            out_mv_channels=mv_channels,
            in_s_channels=s_channels,
            out_s_channels=s_channels,
            output_init="small",
            dropout_prob=dropout_prob,
        )
        self.attention = SelfAttention(basis_q, basis_k, basis_pin, attention)

        # MLP
        mlp = replace(
            mlp,
            mv_channels=(mv_channels, 2 * mv_channels, mv_channels),
            s_channels=(s_channels, 2 * s_channels, s_channels),
            dropout_prob=dropout_prob,
        )
        self.mlp = GeoMLP(basis_pin, mlp, gp, outer)

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor,
        reference_mv=None,
        additional_qk_features_mv=None,
        additional_qk_features_s=None,
        attention_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: norm → attn → + → norm → MLP → +"""

        # Attention block
        h_mv, h_s = self.norm(multivectors, scalars=scalars)
        h_mv, h_s = self.attention(
            h_mv, scalars=h_s,
            additional_qk_features_mv=additional_qk_features_mv,
            additional_qk_features_s=additional_qk_features_s,
            attention_mask=attention_mask,
        )
        outputs_mv = multivectors + h_mv
        outputs_s = scalars + h_s

        # MLP block
        h_mv, h_s = self.norm2(outputs_mv, scalars=outputs_s)
        h_mv, h_s = self.mlp(h_mv, scalars=h_s, reference_mv=reference_mv)
        outputs_mv = outputs_mv + h_mv
        outputs_s = outputs_s + h_s

        return outputs_mv, outputs_s
