"""Equivariant normalization layers for CGA Cl(4,1).

M1 baked in: grade-wise equivariant LayerNorm is now the permanent default.
The ablation env flag has been removed; gradewise=True is always used.
"""

from typing import Tuple

import torch
from torch import nn

from src.cgatr.primitives.normalization import equi_layer_norm


class EquiLayerNorm(nn.Module):
    """Equivariant LayerNorm for CGA multivectors.

    Parameters
    ----------
    ip_weights : torch.Tensor with shape (32,)
        Inner product weights for CGA.
    hidden_s_channels : int
        Number of scalar channels (for standard LayerNorm).
    mv_channel_dim : int
    epsilon : float
    """

    def __init__(
        self,
        ip_weights,
        hidden_s_channels: int = 64,
        mv_channel_dim=-2,
        scalar_channel_dim=-1,
        epsilon: float = 0.01,
    ):
        super().__init__()
        self.mv_channel_dim = mv_channel_dim
        self.epsilon = epsilon
        self.register_buffer("ip_weights", ip_weights)
        self.hidden_s_channels = hidden_s_channels
        # M1: grade-wise norm is permanently enabled (de Haan 2311.04744 Sec 3.4).
        # Avoids null multivectors from +/- grade cancellation that otherwise
        # let coefficients grow by 1/sqrt(epsilon) each layer.

    def forward(
        self, multivectors: torch.Tensor, scalars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., channels, 32)
        scalars : torch.Tensor with shape (..., s_channels)

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., channels, 32)
        output_scalars : torch.Tensor with shape (..., s_channels)
        """
        outputs_mv = equi_layer_norm(
            self.ip_weights,
            multivectors,
            channel_dim=self.mv_channel_dim,
            epsilon=self.epsilon,
            gradewise=True,  # M1: permanently enabled
        )
        outputs_s = torch.nn.functional.layer_norm(
            scalars, normalized_shape=[self.hidden_s_channels]
        )
        return outputs_mv, outputs_s
