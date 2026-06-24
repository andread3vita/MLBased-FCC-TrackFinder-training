"""QKV computation for CGA self-attention with 32-dim multivectors."""

import torch
from einops import rearrange
from torch import nn

from src.cgatr.layers.attention.config import SelfAttentionConfig
from src.cgatr.layers.linear import EquiLinear


class MultiQueryQKVModule(nn.Module):
    """Multi-query QKV: separate Q per head, shared K/V across heads.

    Parameters
    ----------
    basis_pin : torch.Tensor (9, 32, 32)
    config : SelfAttentionConfig
    """

    def __init__(self, basis_pin, config: SelfAttentionConfig):
        super().__init__()

        self.q_linear = EquiLinear(
            basis_pin=basis_pin,
            in_mv_channels=config.in_mv_channels + config.additional_qk_mv_channels,
            out_mv_channels=config.hidden_mv_channels * config.num_heads,
            in_s_channels=config.in_s_channels + config.additional_qk_s_channels,
            out_s_channels=config.hidden_s_channels * config.num_heads,
        )

        self.k_linear = EquiLinear(
            basis_pin=basis_pin,
            in_mv_channels=config.in_mv_channels + config.additional_qk_mv_channels,
            out_mv_channels=config.hidden_mv_channels,
            in_s_channels=config.in_s_channels + config.additional_qk_s_channels,
            out_s_channels=config.hidden_s_channels,
        )

        self.v_linear = EquiLinear(
            basis_pin=basis_pin,
            in_mv_channels=config.in_mv_channels,
            out_mv_channels=config.hidden_mv_channels,
            in_s_channels=config.in_s_channels,
            out_s_channels=config.hidden_s_channels,
        )

        self.config = config

    def forward(self, inputs, scalars, additional_qk_features_mv=None, additional_qk_features_s=None):
        """Forward pass.

        Returns
        -------
        q_mv, k_mv, v_mv : Tensor with shapes including heads dimension
        q_s, k_s, v_s : Tensor
        """
        if additional_qk_features_mv is not None:
            qk_inputs = torch.cat((inputs, additional_qk_features_mv), dim=-2)
        else:
            qk_inputs = inputs
        if scalars is not None and additional_qk_features_s is not None:
            qk_scalars = torch.cat((scalars, additional_qk_features_s), dim=-1)
        else:
            qk_scalars = scalars

        q_mv, q_s = self.q_linear(qk_inputs, qk_scalars)
        k_mv, k_s = self.k_linear(qk_inputs, qk_scalars)
        v_mv, v_s = self.v_linear(inputs, scalars)

        # Rearrange Q to (..., heads, items, channels, 32)
        q_mv = rearrange(
            q_mv,
            "... items (hidden_channels num_heads) x -> ... num_heads items hidden_channels x",
            num_heads=self.config.num_heads,
            hidden_channels=self.config.hidden_mv_channels,
        )
        k_mv = rearrange(k_mv, "... items hidden_channels x -> ... 1 items hidden_channels x")
        v_mv = rearrange(v_mv, "... items hidden_channels x -> ... 1 items hidden_channels x")

        if q_s is not None:
            q_s = rearrange(
                q_s,
                "... items (hidden_channels num_heads) -> ... num_heads items hidden_channels",
                num_heads=self.config.num_heads,
                hidden_channels=self.config.hidden_s_channels,
            )
            k_s = rearrange(k_s, "... items hidden_channels -> ... 1 items hidden_channels")
            v_s = rearrange(v_s, "... items hidden_channels -> ... 1 items hidden_channels")
        else:
            q_s, k_s, v_s = None, None, None

        return q_mv, k_mv, v_mv, q_s, k_s, v_s
