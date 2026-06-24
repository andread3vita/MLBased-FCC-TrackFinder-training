"""Scalar embedding for CGA Cl(4,1) with 32-component multivectors."""

import torch


def embed_scalar(scalars: torch.Tensor) -> torch.Tensor:
    """Embeds scalars into the grade-0 component of 32-dim multivectors.

    Parameters
    ----------
    scalars : torch.Tensor with shape (..., 1)

    Returns
    -------
    multivectors : torch.Tensor with shape (..., 32)
    """
    non_scalar_shape = list(scalars.shape[:-1]) + [31]
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
