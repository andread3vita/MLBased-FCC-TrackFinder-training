# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Factory functions for simple MLPs for multivector data."""

from typing import List, Tuple, Union

import torch
from torch import nn

from src.gatr_v111.layers.dropout import GradeDropout
from src.gatr_v111.layers.linear import EquiLinear
from src.gatr_v111.layers.mlp.config import MLPConfig
from src.gatr_v111.layers.mlp.geometric_bilinears import GeometricBilinear
from src.gatr_v111.layers.mlp.nonlinearities import ScalarGatedNonlinearity


class GeoMLP(nn.Module):
    """Geometric MLP.

    This is a core component of GATr's transformer blocks. It is similar to a regular MLP, except
    that it uses geometric bilinears (GP and equivariant join) in place of the first linear layer.

    Assumes input has shape `(..., channels[0], 16)`, output has shape `(..., channels[-1], 16)`,
    will create hidden layers with shape `(..., channel, 16)` for each additional entry in
    `channels`.

    Parameters
    ----------
    config: MLPConfig
        Configuration object
    """

    def __init__(
        self, basis_pin, config: MLPConfig, gp, outer
    ) -> None:
        super().__init__()

        # Store settings
        self.config = config
        self.basis_pin = basis_pin
        self.gp = gp
        self.outer = outer
        assert config.mv_channels is not None
        s_channels = (
            [None for _ in config.mv_channels] if config.s_channels is None else config.s_channels
        )

        layers: List[nn.Module] = []

        if len(config.mv_channels) >= 2:
            layers.append(
                GeometricBilinear(
                    basis_pin=self.basis_pin,
                    gp=self.gp,
                    outer=self.outer,
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
        """Forward pass.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., in_mv_channels, 16)
            Input multivectors.
        scalars : None or torch.Tensor with shape (..., in_s_channels)
            Optional input scalars.
        reference_mv : torch.Tensor with shape (..., 16)
            Reference multivector for equivariant join.

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., out_mv_channels, 16)
            Output multivectors.
        outputs_s : None or torch.Tensor with shape (..., out_s_channels)
            Output scalars, if scalars are provided. Otherwise None.
        """

        mv, s = multivectors, scalars

        for i, layer in enumerate(self.layers):
            if i == 0:
                mv, s = layer(mv, scalars=s, reference_mv=reference_mv)
                # print("0", mv.shape, s.shape)
                # torch.Size([271, 32, 16]) torch.Size([271, 128])
                # mv = torch.zeros((10,32,16))
                # s = torch.zeros((10,128))
            else :
                mv, s = layer(mv, scalars=s)
            #     # mv = torch.zeros((10,32,16))
            #     # s = torch.zeros((10,128))
            # elif i ==2:
            #     mv, s = layer(mv, scalars=s)
            
                # print(i, mv.shape, s.shape)
                # 1 torch.Size([271, 32, 16]) torch.Size([271, 128])
                # 2 torch.Size([271, 16, 16]) torch.Size([271, 64]

        return mv, s
