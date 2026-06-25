"""Pin(4,1)-equivariant linear maps for CGA Cl(4,1).

The Pin-equivariant linear basis for Cl(4,1) has 9 elements:
- 6 grade projections (grades 0-5, diagonal blocks)
- 3 Hodge-dual cross-grade maps: grade 0<->5, grade 1<->4, grade 2<->3

Grade index ranges for 32-component multivectors:
  Grade 0: [0]         (1 component)
  Grade 1: [1..5]      (5 components)
  Grade 2: [6..15]     (10 components)
  Grade 3: [16..25]    (10 components)
  Grade 4: [26..30]    (5 components)
  Grade 5: [31]        (1 component)
"""

import torch

# Grade ranges for Cl(4,1)
GRADE_RANGES = {
    0: (0, 1),
    1: (1, 6),
    2: (6, 16),
    3: (16, 26),
    4: (26, 31),
    5: (31, 32),
}
GRADE_DIMS = [1, 5, 10, 10, 5, 1]
NUM_MV_COMPONENTS = 32
NUM_GRADES = 6


def _compute_pin_equi_linear_basis(
    device=torch.device("cpu"), dtype=torch.float32, normalize=True
) -> torch.Tensor:
    """Constructs basis elements for Pin(4,1)-equivariant linear maps between multivectors.

    9 basis elements, each a (32, 32) matrix:
    - Elements 0-5: grade projections (identity on grade g, zero elsewhere)
    - Elements 6-8: Hodge-dual cross-grade maps (grade 0<->5, 1<->4, 2<->3)

    Parameters
    ----------
    device : torch.device
    dtype : torch.dtype
    normalize : bool
        Whether to normalize basis elements by Frobenius norm.

    Returns
    -------
    basis : torch.Tensor with shape (9, 32, 32)
    """
    basis = []

    # 6 grade projections
    for grade in range(NUM_GRADES):
        w = torch.zeros((NUM_MV_COMPONENTS, NUM_MV_COMPONENTS))
        start, end = GRADE_RANGES[grade]
        for i in range(start, end):
            w[i, i] = 1.0
        if normalize:
            w /= torch.linalg.norm(w)
        basis.append(w.unsqueeze(0))

    # 3 Hodge-dual cross-grade maps
    # These map between complementary grades: (0<->5), (1<->4), (2<->3)
    # For non-degenerate Cl(4,1), the Hodge dual maps grade-k to grade-(5-k)
    # with the same number of components
    dual_pairs = [(0, 5), (1, 4), (2, 3)]
    for grade_low, grade_high in dual_pairs:
        w = torch.zeros((NUM_MV_COMPONENTS, NUM_MV_COMPONENTS))
        start_low, end_low = GRADE_RANGES[grade_low]
        start_high, end_high = GRADE_RANGES[grade_high]
        dim = end_low - start_low
        assert dim == end_high - start_high, (
            f"Grade {grade_low} and {grade_high} must have same dimension"
        )
        # Map component i of grade_low to component i of grade_high (and vice versa)
        for i in range(dim):
            w[start_high + i, start_low + i] = 1.0
            w[start_low + i, start_high + i] = 1.0
        if normalize:
            w /= torch.linalg.norm(w)
        basis.append(w.unsqueeze(0))

    catted_basis = torch.cat(basis, dim=0)
    return catted_basis.to(device=device, dtype=dtype)


def _compute_se3_equi_linear_basis(
    gp: torch.Tensor,
    device=torch.device("cpu"),
    dtype=torch.float32,
    tol: float = 1e-6,
    verify: bool = True,
) -> torch.Tensor:
    """SE(3)-equivariant linear-map basis for CGA Cl(4,1).

    Computed as the null space of the Lie-algebra equivariance constraint — the
    numerical construction of de Haan et al. 2311.04744, Sec. 3.3 — rather than
    hand-listed maps. This is the *correct* C-GATr linear layer: its span is
    exactly the grade projections plus the e_inf (point-at-infinity)
    multiplication maps the paper derives, with no ad-hoc Hodge cross-grade maps.

    The generators of SE(3) ⊂ Spin(4,1) are the six bivectors
        e1∧e2, e1∧e3, e2∧e3        (rotations)
        e1∧∞,  e2∧∞,  e3∧∞         (translations,  ∞ = e+ + e-)
    A linear map φ : Cl→Cl (a 32×32 matrix) is equivariant iff it commutes with
    the algebra action dρ(B)·x = B x − x B of every generator B. We stack those
    commutator constraints over all six generators and return an orthonormal
    (Frobenius) basis of their null space.

    We deliberately omit the extra mirror/parity constraint: a solenoidal B-field
    makes track reconstruction chiral, so reflection equivariance is physically
    inappropriate. The result is the SE(3) (proper rigid-motion) basis — a
    superset of the full-E(3) one and the right inductive bias here.

    Parameters
    ----------
    gp : torch.Tensor (32, 32, 32)
        Geometric-product Cayley table.
    verify : bool
        If True, assert every returned map commutes with every generator.

    Returns
    -------
    basis : torch.Tensor (n_basis, 32, 32)
    """
    n = NUM_MV_COMPONENTS
    # Solve in float64 for a clean, well-separated null space.
    gp64 = gp.to(device=device, dtype=torch.float64)

    def _unit(idx: int) -> torch.Tensor:
        v = torch.zeros(n, dtype=torch.float64, device=device)
        v[idx] = 1.0
        return v

    e1, e2, e3 = _unit(1), _unit(2), _unit(3)
    e_inf = _unit(4) + _unit(5)  # point at infinity ∞ = e+ + e-

    def _gp_vec(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # geometric product of two multivectors: out_i = Σ_jk gp[i,j,k] a_j b_k
        return torch.einsum("ijk,j,k->i", gp64, a, b)

    generators = [
        _gp_vec(e1, e2), _gp_vec(e1, e3), _gp_vec(e2, e3),        # so(3) rotations
        _gp_vec(e1, e_inf), _gp_vec(e2, e_inf), _gp_vec(e3, e_inf),  # translations
    ]

    eye = torch.eye(n, dtype=torch.float64, device=device)
    gen_matrices = []
    # Accumulate the normal matrix G = Σ_B C_Bᵀ C_B (1024×1024). Its null space
    # equals the common null space of all per-generator constraints C_B, but is
    # far cheaper to factor than stacking them into a (6·1024, 1024) matrix.
    G = torch.zeros(n * n, n * n, dtype=torch.float64, device=device)
    for B in generators:
        # dρ(B): x ↦ B x − x B, as a 32×32 matrix M.
        m_left = torch.einsum("ikj,k->ij", gp64, B)   # (B x)_i = Σ_k gp[i,k,j] B_k
        m_right = torch.einsum("ijk,k->ij", gp64, B)  # (x B)_i = Σ_k gp[i,j,k] B_k
        M = m_left - m_right
        gen_matrices.append(M)
        # Operator on vec(φ): C(φ)[i,j] = Σ_kl T[i,j,k,l] φ[k,l] = (Mφ − φM)[i,j].
        T = (torch.einsum("ik,lj->ijkl", M, eye)
             - torch.einsum("ik,lj->ijkl", eye, M))
        C = T.reshape(n * n, n * n)
        G = G + C.T @ C

    evals, evecs = torch.linalg.eigh(G)  # ascending eigenvalues; orthonormal cols
    thresh = tol * (evals.abs().max() if evals.numel() else torch.tensor(1.0))
    null_mask = evals < thresh
    basis = evecs[:, null_mask].T.reshape(-1, n, n)  # (n_basis, 32, 32), orthonormal

    if verify:
        if basis.shape[0] == 0:
            raise RuntimeError(
                "SE(3)-equivariant CGA basis is empty — check the GP Cayley table."
            )
        for M in gen_matrices:
            resid = (torch.einsum("ik,bkj->bij", M, basis)
                     - torch.einsum("bik,kj->bij", basis, M))
            max_resid = resid.abs().max().item()
            if max_resid > 1e-6:
                raise RuntimeError(
                    f"CGA equivariance self-check failed: "
                    f"max||[M_B, phi]|| = {max_resid:.2e} (expected ~0)."
                )
        print(
            f"[cgatr] SE(3)-equivariant CGA linear basis: {basis.shape[0]} maps "
            f"(verified, max residual {max_resid:.1e})",
            flush=True,
        )

    return basis.to(dtype=dtype)


def _compute_reversal(device=torch.device("cpu"), dtype=torch.float32) -> torch.Tensor:
    """Reversal signs for Cl(4,1): grade k gets sign (-1)^{k(k-1)/2}.

    Grade 0: +1, Grade 1: +1, Grade 2: -1, Grade 3: -1, Grade 4: +1, Grade 5: +1

    Returns
    -------
    reversal_diag : torch.Tensor with shape (32,)
    """
    reversal = torch.ones(NUM_MV_COMPONENTS, device=device, dtype=dtype)
    # Grade 2: indices 6..15 -> sign = -1
    reversal[6:16] = -1
    # Grade 3: indices 16..25 -> sign = -1
    reversal[16:26] = -1
    return reversal


def _compute_grade_involution(device=torch.device("cpu"), dtype=torch.float32) -> torch.Tensor:
    """Grade involution signs: grade k gets sign (-1)^k.

    Grade 0: +1, Grade 1: -1, Grade 2: +1, Grade 3: -1, Grade 4: +1, Grade 5: -1

    Returns
    -------
    involution_diag : torch.Tensor with shape (32,)
    """
    involution = torch.ones(NUM_MV_COMPONENTS, device=device, dtype=dtype)
    # Odd grades: 1 (indices 1..5), 3 (indices 16..25), 5 (index 31)
    involution[1:6] = -1
    involution[16:26] = -1
    involution[31] = -1
    return involution


NUM_PIN_LINEAR_BASIS_ELEMENTS = 9


def equi_linear(basis: torch.Tensor, x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """Pin(4,1)-equivariant linear map.

    f(x) = sum_{a} coeffs_{y,x,a} * basis_{a,i,j} * x_{..., x, j}

    Parameters
    ----------
    basis : torch.Tensor with shape (9, 32, 32)
    x : torch.Tensor with shape (..., x_channels, 32)
        Arbitrary leading dims are supported (e.g. `(items, x_channels, 32)`
        for single-event, `(B, N, x_channels, 32)` for batched).
    coeffs : torch.Tensor with shape (y_channels, x_channels, 9)

    Returns
    -------
    result : torch.Tensor with shape (..., y_channels, 32)
        Same leading dims as `x`.
    """
    # NOTE: We deliberately don't assert `coeffs.shape[-1] == basis.shape[0]`
    # here. That sanity check is structurally guaranteed by construction —
    # `coeffs` is `nn.Parameter(empty(y, x, NUM_PIN_LINEAR_BASIS_ELEMENTS))`
    # and `basis` is the precomputed pin-equivariant basis with the same
    # leading dim. During tracing, the assertion would convert two
    # SymInts to a Python bool and emit a TracerWarning. The ONNX
    # exporter would happily bake the result in as a constant, but
    # there's no upside.
    y, x_dim, a = coeffs.shape
    a_, i, j = basis.shape

    # coeffs @ basis -> (y, x, i, j)
    coeffs_flat = coeffs.reshape(-1, a)              # (y*x, a)
    basis_flat = basis.reshape(a, -1)                # (a, i*j)
    c2_flat = torch.matmul(coeffs_flat, basis_flat)  # (y*x, i*j)
    c2 = c2_flat.view(y, x_dim, i, j)                # (y, x, i, j)
    c2 = c2.permute(0, 1, 3, 2)                      # (y, x, j, i)

    # Ellipsis lets `x` carry any leading-dim shape: 3-D (items, x, 32)
    # for single-event inference, 4-D (B, N, x, 32) for multi-event
    # batched ONNX, etc.
    result = torch.einsum("...xj,yxji->...yi", x, c2)
    return result


def grade_project(x: torch.Tensor) -> torch.Tensor:
    """Projects input to individual grades.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 32)

    Returns
    -------
    outputs : torch.Tensor with shape (..., 6, 32)
        The second-to-last dimension indexes the grades (0-5).
    """
    basis = _compute_pin_equi_linear_basis(device=x.device, dtype=x.dtype, normalize=False)
    # First 6 basis elements are grade projections
    basis = basis[:6]
    projections = torch.einsum("g i j, ... j -> ... g i", basis, x)
    return projections


def reverse(x: torch.Tensor) -> torch.Tensor:
    """Reversal of a multivector: flips sign for grades 2 and 3."""
    return _compute_reversal(device=x.device, dtype=x.dtype) * x


def grade_involute(x: torch.Tensor) -> torch.Tensor:
    """Grade involution: flips sign for odd grades."""
    return _compute_grade_involution(device=x.device, dtype=x.dtype) * x
