"""Grade dropout for CGA Cl(4,1) with 6 grades."""

import torch

from src.cgatr.primitives.linear import grade_project


def grade_dropout(x: torch.Tensor, p: float, training: bool = True) -> torch.Tensor:
    """Multivector dropout, dropping out grades independently.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 32)
    p : float
        Dropout probability per grade.
    training : bool

    Returns
    -------
    outputs : torch.Tensor with shape (..., 32)
    """
    # Project to 6 grades
    x = grade_project(x)  # (..., 6, 32)

    # Apply dropout over grade dimension
    h = x.view(-1, 6, 32)
    h = torch.nn.functional.dropout1d(h, p=p, training=training, inplace=False)
    h = h.view(x.shape)

    # Combine grades
    h = torch.sum(h, dim=-2)
    return h
