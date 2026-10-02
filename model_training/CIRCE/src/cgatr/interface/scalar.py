"""Scalar embedding for CGA Cl(4,1) with 32-component multivectors."""

import torch


def embed_scalar(scalars: torch.Tensor, num_blades: int = 32) -> torch.Tensor:
    """Embeds scalars into the grade-0 component of multivectors.

    Parameters
    ----------
    scalars : torch.Tensor with shape (..., 1)
    num_blades : int
        Width of the algebra: 32 for the conformal Cl(4,1), 16 for the
        projective Cl(3,0,1).

    Returns
    -------
    multivectors : torch.Tensor with shape (..., num_blades)
    """
    non_scalar_shape = list(scalars.shape[:-1]) + [num_blades - 1]
    non_scalar_components = torch.zeros(
        non_scalar_shape, device=scalars.device, dtype=scalars.dtype
    )
    return torch.cat((scalars, non_scalar_components), dim=-1)


def extract_scalar(multivectors: torch.Tensor) -> torch.Tensor:
    """Extracts grade-0 (scalar) component from 32-dim multivectors.

    Parameters
    ----------
    multivectors : torch.Tensor with shape (..., 32)

    Returns
    -------
    scalars : torch.Tensor with shape (..., 1)
    """
    return multivectors[..., [0]]
