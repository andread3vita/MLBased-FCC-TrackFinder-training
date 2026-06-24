# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
# from functools import lru_cache

import torch

from src.gatr_v111.utils.einsum import cached_einsum, custom_einsum


def _compute_pin_equi_linear_basis(
    device=torch.device("cpu"), dtype=torch.float32, normalize=True
) -> torch.Tensor:
    """Constructs basis elements for Pin(3,0,1)-equivariant linear maps between multivectors.

    This function is cached.

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype
    normalize : bool
        Whether to normalize the basis elements

    Returns
    -------
    basis : torch.Tensor with shape (7, 16, 16)
        Basis elements for equivariant linear maps.
    """

    # We constructed these manually in a notebook, here hardcoded for convenience
    basis_elements = [
        [0],
        [1, 2, 3, 4],
        [5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14],
        [15],
        [(1, 0)],
        [(5, 2), (6, 3), (7, 4)],
        [(11, 8), (12, 9), (13, 10)],
        [(15, 14)],
    ]
    basis = []

    for elements in basis_elements:
        w = torch.zeros((16, 16))
        for element in elements:
            if isinstance(element, tuple):
                i, j = element
                w[i, j] = 1.0
            else:
                w[element, element] = 1.0

        if normalize:
            w /= torch.linalg.norm(w)

        w = w.unsqueeze(0)
        basis.append(w)

    catted_basis = torch.cat(basis, dim=0)

    return catted_basis.to(device=device, dtype=dtype)




# @lru_cache()
def _compute_reversal(device=torch.device("cpu"), dtype=torch.float32) -> torch.Tensor:
    """Constructs a matrix that computes multivector reversal.

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype

    Returns
    -------
    reversal_diag : torch.Tensor with shape (16,)
        The diagonal of the reversal matrix, consisting of +1 and -1 entries.
    """
    reversal_flat = torch.ones(16, device=device, dtype=dtype)
    reversal_flat[5:15] = -1
    return reversal_flat


# @lru_cache()
def _compute_grade_involution(device=torch.device("cpu"), dtype=torch.float32) -> torch.Tensor:
    """Constructs a matrix that computes multivector grade involution.

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype

    Returns
    -------
    involution_diag : torch.Tensor with shape (16,)
        The diagonal of the involution matrix, consisting of +1 and -1 entries.
    """
    involution_flat = torch.ones(16, device=device, dtype=dtype)
    involution_flat[1:5] = -1
    involution_flat[11:15] = -1
    return involution_flat


NUM_PIN_LINEAR_BASIS_ELEMENTS = len(_compute_pin_equi_linear_basis())


# def equi_linear(basis, x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
#     """Pin-equivariant linear map f(x) = sum_{a,j} coeffs_a W^a_ij x_j.

#     The W^a are seven pre-defined basis elements.

#     Parameters
#     ----------
#     x : torch.Tensor with shape (..., in_channels, 16)
#         Input multivector. Batch dimensions must be broadcastable between x and coeffs.
#     coeffs : torch.Tensor with shape (out_channels, in_channels, 7)
#         Coefficients for the 59 basis elements. Batch dimensions must be broadcastable between x and
#         coeffs.

#     Returns
#     -------
#     outputs : torch.Tensor with shape (..., 16)
#         Result. Batch dimensions are result of broadcasting between x and coeffs.
#     """
#     # basis = _compute_pin_equi_linear_basis(device=x.device, dtype=x.dtype)
#     # return custom_einsum("y x a, a i j, ... x j -> ... y i", coeffs, basis, x, path=[0, 1, 0, 1])
#     c2 = torch.einsum("y x a, a i j ->  y x i j ", coeffs, basis)
#     a2 = torch.einsum("y x i j , l x j -> l y i", c2, x)
#     return a2


def equi_linear(basis: torch.Tensor, x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    # coeffs: (y, x, a)
    # basis: (a, i, j)
    # x:     (l, x, j)

    y, x_dim, a = coeffs.shape
    a_, i, j = basis.shape
    l = x.shape[0]
    assert a == a_

    # coeffs @ basis → (y, x, i, j)
    coeffs_flat = coeffs.reshape(-1, a)         # (y*x, a)
    basis_flat = basis.reshape(a, -1)           # (a, i*j)
    c2_flat = torch.matmul(coeffs_flat, basis_flat)  # (y*x, i*j)
    c2 = c2_flat.view(y, x_dim, i, j)           # (y, x, i, j)
    c2 = c2.permute(0, 1, 3, 2)                 # (y, x, j, i)

    # x: (l, x, j)
    # c2: (y, x, j, i)
    # Ziel: (l, y, i) ← über x, j summieren

    result = torch.einsum("lxj,yxji->lyi", x, c2)  # (l, y, i)
    return result

def grade_project(x: torch.Tensor) -> torch.Tensor:
    """Projects an input tensor to the individual grades.

    The return value is a single tensor with a new grade dimension.

    NOTE: this primitive is not used widely in our architectures.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 5, 16)
        Output multivector. The second-to-last dimension indexes the grades.
    """

    # Select kernel on correct device
    basis = _compute_pin_equi_linear_basis(device=x.device, dtype=x.dtype, normalize=False)

    # First five basis elements are grade projections
    basis = basis[:5]

    # Project to grades
    # projections = cached_einsum("g i j, ... j -> ... g i", basis, x)
    projections = torch.einsum("g i j, ... j -> ... g i", basis, x)

    return projections


def reverse(x: torch.Tensor) -> torch.Tensor:
    """Computes the reversal of a multivector.

    The reversal has the same scalar, vector, and pseudoscalar components, but flips sign in the
    bivector and trivector components.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Output multivector.
    """
    return _compute_reversal(device=x.device, dtype=x.dtype) * x


def grade_involute(x: torch.Tensor) -> torch.Tensor:
    """Computes the grade involution of a multivector.

    The reversal has the same scalar, bivector, and pseudoscalar components, but flips sign in the
    vector and trivector components.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Output multivector.
    """

    return _compute_grade_involution(device=x.device, dtype=x.dtype) * x
