"""Scalar-gated nonlinearities for CGA — dimension-agnostic."""

from typing import Tuple

import torch
from torch import nn

from src.cgatr.primitives.nonlinearities import gated_gelu, gated_relu, gated_sigmoid


class ScalarGatedNonlinearity(nn.Module):
    """Gated nonlinearity using the scalar (grade-0) component as gate.

    For 32-dim CGA multivectors, gate = mv[..., [0]] (same as PGA).

    Parameters
    ----------
    nonlinearity : {"relu", "sigmoid", "gelu"}
    """

    def __init__(self, nonlinearity: str = "relu", **kwargs) -> None:
        super().__init__()
        gated_fn_dict = dict(relu=gated_relu, gelu=gated_gelu, sigmoid=gated_sigmoid)
        scalar_fn_dict = dict(
            relu=nn.functional.relu, gelu=nn.functional.gelu, sigmoid=nn.functional.sigmoid
        )
        self.gated_nonlinearity = gated_fn_dict[nonlinearity]
        self.scalar_nonlinearity = scalar_fn_dict[nonlinearity]

    def forward(
        self, multivectors: torch.Tensor, scalars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gates = multivectors[..., [0]]
        outputs_mv = self.gated_nonlinearity(multivectors, gates=gates)
        outputs_s = self.scalar_nonlinearity(scalars)
        return outputs_mv, outputs_s
