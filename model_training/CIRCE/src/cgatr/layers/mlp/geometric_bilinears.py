"""Geometric bilinear layer for CGA: geometric product only.

de Haan et al. 2311.04744 (Prop. 1) prove that for a *non-degenerate* algebra
such as the CGA, any equivariant multilinear map can be built from the geometric
product alone — the join is only needed for the (degenerate) PGA. The previous
implementation also carried a join branch whose reference multivector (the mean
of the grade-{0,1,2} input embeddings) had an identically-zero pseudoscalar
component, so it contributed nothing but fed zeros into `linear_out`. We drop it
and let the geometric-product branch use the full hidden width.
"""

import os
from typing import Optional, Tuple

import torch
from torch import nn

from src.cgatr.layers.linear import EquiLinear, fused_equi_linear_same_input
from src.cgatr.primitives import geometric_product
from src.cgatr.primitives.dual import equivariant_join


class GeometricBilinear(nn.Module):
    """CGA geometric bilinear: a single geometric-product interaction.

    Parameters
    ----------
    basis_pin : (n_basis, 32, 32)
    gp : (32, 32, 32) GP table
    outer : (32, 32, 32) outer product table (unused; kept for signature compat)
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
        use_join: bool = False,
        pseudoscalar_idx: int = 31,
    ) -> None:
        super().__init__()
        self.register_buffer("_gp", gp)

        if hidden_mv_channels is None:
            hidden_mv_channels = out_mv_channels

        # The join is required in a degenerate algebra, where the geometric
        # product alone no longer spans the equivariant multilinear maps. With it
        # enabled the hidden width is split between the two branches, so
        # `linear_out` sees the same width either way.
        self.use_join = use_join
        if use_join:
            if hidden_mv_channels % 2:
                raise ValueError(
                    f"hidden_mv_channels must be even with use_join, got "
                    f"{hidden_mv_channels}"
                )
            branch_channels = hidden_mv_channels // 2
        else:
            branch_channels = hidden_mv_channels

        # GP branch — full hidden width unless the join takes half.
        self.linear_left = EquiLinear(
            basis_pin, in_mv_channels, branch_channels,
            in_s_channels=in_s_channels, out_s_channels=None,
        )
        self.linear_right = EquiLinear(
            basis_pin, in_mv_channels, branch_channels,
            in_s_channels=in_s_channels, out_s_channels=None,
            initialization="almost_unit_scalar",
        )

        # Output projection
        self.linear_out = EquiLinear(
            basis_pin, hidden_mv_channels, out_mv_channels, in_s_channels, out_s_channels
        )

        self.geometric_product = geometric_product(self._gp)
        self.join = (equivariant_join(outer, pseudoscalar_idx=pseudoscalar_idx)
                     if use_join else None)
        fused = os.environ.get("CGATR_FUSED_PROJECTIONS", "").strip().lower()
        self._use_fused = fused in ("1", "true", "yes")

    def forward(
        self,
        multivectors: torch.Tensor,
        reference_mv: torch.Tensor = None,
        scalars: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        multivectors : (..., in_mv_channels, 32)
        reference_mv : unused (kept so GeoMLP's call signature is unchanged)
        scalars : (..., in_s_channels)

        Returns
        -------
        outputs_mv : (..., out_mv_channels, 32)
        outputs_s : (..., out_s_channels) or None
        """
        if self._use_fused and scalars is not None:
            (left, _), (right, _) = fused_equi_linear_same_input(
                (self.linear_left, self.linear_right),
                multivectors,
                scalars,
            )
        else:
            left, _ = self.linear_left(multivectors, scalars=scalars)
            right, _ = self.linear_right(multivectors, scalars=scalars)
        hidden = self.geometric_product(left, right)

        if self.join is not None:
            hidden = torch.cat([hidden, self.join(left, right)], dim=-2)

        outputs_mv, outputs_s = self.linear_out(hidden, scalars=scalars)
        return outputs_mv, outputs_s
