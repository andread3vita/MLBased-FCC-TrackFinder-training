"""Equivariant dropout for CGA Cl(4,1)."""

from typing import Tuple

import torch
from torch import nn

from src.cgatr.primitives import grade_dropout


class GradeDropout(nn.Module):
    """Grade dropout for 32-dim multivectors (6 grades) + standard dropout for scalars.

    Parameters
    ----------
    p : float
        Dropout probability.
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        self._dropout_prob = p

    def forward(
        self, multivectors: torch.Tensor, scalars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out_mv = grade_dropout(multivectors, p=self._dropout_prob, training=self.training)
        out_s = torch.nn.functional.dropout(scalars, p=self._dropout_prob, training=self.training)
        return out_mv, out_s
