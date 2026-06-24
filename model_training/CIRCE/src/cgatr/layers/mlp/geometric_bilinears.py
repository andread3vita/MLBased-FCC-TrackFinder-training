"""Geometric bilinear layer for CGA: GP + equivariant join."""

from typing import Optional, Tuple

import torch
from torch import nn

from src.cgatr.layers.linear import EquiLinear
from src.cgatr.primitives import equivariant_join, geometric_product


class GeometricBilinear(nn.Module):
    """CGA geometric bilinear: GP branch + join branch, each producing half the channels.

    Parameters
    ----------
    basis_pin : (9, 32, 32)
    gp : (32, 32, 32) GP table
    outer : (32, 32, 32) outer product table
    in_mv_channels : int
    out_mv_channels : int
    hidden_mv_channels : int or None
    in_s_channels : int or None
    out_s_channels : int or None
    """

    def __init__(
        self,
        basis_pin,
        gp,
        outer,
        in_mv_channels: int,
        out_mv_channels: int,
        hidden_mv_channels: Optional[int] = None,
        in_s_channels: Optional[int] = None,
        out_s_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("_gp", gp)
        self.register_buffer("_outer", outer)

        if hidden_mv_channels is None:
            hidden_mv_channels = out_mv_channels

        out_mv_channels_each = hidden_mv_channels // 2
        assert out_mv_channels_each * 2 == hidden_mv_channels, "Need even channel count"

        # GP branch
        self.linear_left = EquiLinear(
            basis_pin, in_mv_channels, out_mv_channels_each,
            in_s_channels=in_s_channels, out_s_channels=None,
        )
        self.linear_right = EquiLinear(
            basis_pin, in_mv_channels, out_mv_channels_each,
            in_s_channels=in_s_channels, out_s_channels=None,
            initialization="almost_unit_scalar",
        )

        # Join branch
        self.linear_join_left = EquiLinear(
            basis_pin, in_mv_channels, out_mv_channels_each,
            in_s_channels=in_s_channels, out_s_channels=None,
        )
        self.linear_join_right = EquiLinear(
            basis_pin, in_mv_channels, out_mv_channels_each,
            in_s_channels=in_s_channels, out_s_channels=None,
        )

        # Output projection
        self.linear_out = EquiLinear(
            basis_pin, hidden_mv_channels, out_mv_channels, in_s_channels, out_s_channels
        )

        self.geometric_product = geometric_product(self._gp)
        self.equivariant_join = equivariant_join(self._outer)

    def forward(
        self,
        multivectors: torch.Tensor,
        reference_mv: torch.Tensor,
        scalars: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        multivectors : (..., in_mv_channels, 32)
        reference_mv : (..., 32)
        scalars : (..., in_s_channels)

        Returns
        -------
        outputs_mv : (..., out_mv_channels, 32)
        outputs_s : (..., out_s_channels) or None
        """
        # GP branch
        left, _ = self.linear_left(multivectors, scalars=scalars)
        right, _ = self.linear_right(multivectors, scalars=scalars)
        gp_outputs = self.geometric_product(left, right)

        # Join branch
        left, _ = self.linear_join_left(multivectors, scalars=scalars)
        right, _ = self.linear_join_right(multivectors, scalars=scalars)
        join_outputs = self.equivariant_join(left, right, reference_mv)

        # Combine and project
        outputs_mv = torch.cat((gp_outputs, join_outputs), dim=-2)
        outputs_mv, outputs_s = self.linear_out(outputs_mv, scalars=scalars)

        return outputs_mv, outputs_s
