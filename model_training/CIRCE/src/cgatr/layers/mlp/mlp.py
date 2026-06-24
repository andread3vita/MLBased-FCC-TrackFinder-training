"""Geometric MLP for CGA Cl(4,1)."""

from typing import List, Tuple, Union

import torch
from torch import nn

from src.cgatr.layers.dropout import GradeDropout
from src.cgatr.layers.linear import EquiLinear
from src.cgatr.layers.mlp.config import MLPConfig
from src.cgatr.layers.mlp.geometric_bilinears import GeometricBilinear
from src.cgatr.layers.mlp.nonlinearities import ScalarGatedNonlinearity


class GeoMLP(nn.Module):
    """CGA Geometric MLP.

    GeometricBilinear (GP + join) -> activation -> EquiLinear.
    Typical channel config: (C, 2C, C).

    Parameters
    ----------
    basis_pin : (9, 32, 32)
    config : MLPConfig
    gp : (32, 32, 32)
    outer : (32, 32, 32)
    """

    def __init__(self, basis_pin, config: MLPConfig, gp, outer) -> None:
        super().__init__()
        self.config = config
        assert config.mv_channels is not None
        s_channels = (
            [None for _ in config.mv_channels] if config.s_channels is None else config.s_channels
        )

        layers: List[nn.Module] = []

        if len(config.mv_channels) >= 2:
            layers.append(
                GeometricBilinear(
                    basis_pin=basis_pin,
                    gp=gp,
                    outer=outer,
                    in_mv_channels=config.mv_channels[0],
                    out_mv_channels=config.mv_channels[1],
                    in_s_channels=s_channels[0],
                    out_s_channels=s_channels[1],
                )
            )
            if config.dropout_prob is not None:
                layers.append(GradeDropout(config.dropout_prob))

            for in_, out, in_s, out_s in zip(
                config.mv_channels[1:-1], config.mv_channels[2:], s_channels[1:-1], s_channels[2:]
            ):
                layers.append(ScalarGatedNonlinearity(config.activation))
                layers.append(EquiLinear(basis_pin, in_, out, in_s_channels=in_s, out_s_channels=out_s))
                if config.dropout_prob is not None:
                    layers.append(GradeDropout(config.dropout_prob))

        self.layers = nn.ModuleList(layers)

    def forward(
        self, multivectors: torch.Tensor, scalars: torch.Tensor, reference_mv: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        mv, s = multivectors, scalars
        for i, layer in enumerate(self.layers):
            if i == 0:
                mv, s = layer(mv, scalars=s, reference_mv=reference_mv)
            else:
                mv, s = layer(mv, scalars=s)
        return mv, s
