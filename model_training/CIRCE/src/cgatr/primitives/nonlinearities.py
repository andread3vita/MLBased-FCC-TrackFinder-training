"""Gated nonlinearities for CGA Cl(4,1).

These are dimension-agnostic — identical to PGA since they operate on
the scalar component (index 0) which exists in all Clifford algebras.
"""

import math

import torch

_GATED_GELU_DIV_FACTOR = math.sqrt(2 / math.pi) * 2


def gated_relu(x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Pin-equivariant gated ReLU: ReLU(gates) * x."""
    return torch.nn.functional.relu(gates) * x


def gated_sigmoid(x: torch.Tensor, gates: torch.Tensor):
    """Pin-equivariant gated sigmoid: sigmoid(gates) * x."""
    return torch.nn.functional.sigmoid(gates) * x


def gated_gelu(x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Pin-equivariant gated GeLU: GeLU(gates) * x."""
    return torch.nn.functional.gelu(gates, approximate="tanh") * x


def gated_gelu_divide(x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Pin-equivariant gated GeLU with division."""
    weights = torch.sigmoid(_GATED_GELU_DIV_FACTOR * (gates + 0.044715 * gates**3))
    return weights * x
