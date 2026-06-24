"""Dualization and equivariant join for CGA Cl(4,1).

The dual maps each blade to its complement via the pseudoscalar e12345.
For non-degenerate Cl(4,1), the dual permutation and signs are loaded from
precomputed metadata, or computed on the fly from the GP table.
"""

import torch
from torch import nn

from src.cgatr.primitives.bilinear import outer_product


@torch.no_grad()
def compute_dualization_from_gp(gp: torch.Tensor):
    """Compute dual permutation and signs from the GP table.

    dual(blade_j) = blade_j * I^{-1}

    For Cl(4,1), I = e12345, I^2 = e12345 * e12345.
    We compute I^{-1} from the GP table.

    Parameters
    ----------
    gp : torch.Tensor with shape (32, 32, 32)
        Geometric product Cayley table.

    Returns
    -------
    permutation : list of int
    signs : torch.Tensor with shape (32,)
    """
    n = gp.shape[0]
    # Pseudoscalar is the last blade (index 31)
    ps_idx = n - 1

    # Compute I * I using GP table: result_i = sum_j gp[i, ps_idx, j] * I_j
    # But I only has component at ps_idx, so: (I * I)_i = gp[i, ps_idx, ps_idx]
    I_squared_scalar = gp[0, ps_idx, ps_idx].item()  # Should be scalar (grade 0)

    # I^{-1} = I / (I*I) = I / I_squared_scalar
    # dual(blade_j) = blade_j * I^{-1} = (1/I_sq) * (blade_j * I)
    # (blade_j * I)_i = gp[i, j, ps_idx]

    permutation = []
    signs = torch.zeros(n, dtype=torch.float32)

    for j in range(n):
        # blade_j * I: result_i = gp[i, j, ps_idx]
        result = gp[:, j, ps_idx]
        nonzero = result.nonzero(as_tuple=True)[0]
        assert len(nonzero) == 1, (
            f"Blade {j} dual has {len(nonzero)} nonzero components"
        )
        i = nonzero[0].item()
        permutation.append(i)
        signs[j] = result[i].item() / I_squared_scalar

    return permutation, signs


class _DualCache:
    """Lazy-initialized cache for dualization constants."""

    permutation = None
    signs = None

    @classmethod
    def init_from_metadata(cls, metadata: dict, device=torch.device("cpu")):
        cls.permutation = metadata["dual_permutation"]
        cls.signs = metadata["dual_signs"].to(device=device)

    @classmethod
    def init_from_gp(cls, gp: torch.Tensor):
        cls.permutation, cls.signs = compute_dualization_from_gp(gp)
        cls.signs = cls.signs.to(device=gp.device)


def dual(x: torch.Tensor) -> torch.Tensor:
    """Computes the Hodge dual of a multivector in Cl(4,1).

    Parameters
    ----------
    x : torch.Tensor with shape (..., 32)

    Returns
    -------
    result : torch.Tensor with shape (..., 32)
    """
    assert _DualCache.permutation is not None, (
        "Dual not initialized. Call _DualCache.init_from_metadata() or init_from_gp() first."
    )
    signs = _DualCache.signs.to(device=x.device, dtype=x.dtype)
    return signs * x[..., _DualCache.permutation]


class equivariant_join(nn.Module):
    """Equivariant join: dual(outer_product(dual(x), dual(y))) * reference_pseudoscalar."""

    def __init__(self, outer) -> None:
        super().__init__()
        self.register_buffer("outer", outer)

    def forward(self, x, y, reference):
        # reference[..., [31]] is the pseudoscalar component (grade 5, index 31)
        ref_ps = reference[..., [31]]
        return ref_ps * dual(outer_product(self.outer, dual(x), dual(y)))
