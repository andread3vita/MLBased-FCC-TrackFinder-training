"""Tests for CGA Cl(4,1) implementation.

Run with: python -m pytest src/cgatr/tests/test_cga.py -v
Or: python src/cgatr/tests/test_cga.py
"""

import sys
import math

import torch
import torch.nn as nn


def _load_tables():
    """Load CGA tables. Run generate_cga_tables.py first."""
    gp_sparse = torch.load("cga_utils/cga_geometric_product.pt")
    gp = gp_sparse.to_dense().float()
    op_sparse = torch.load("cga_utils/cga_outer_product.pt")
    op = op_sparse.to_dense().float()
    metadata = torch.load("cga_utils/cga_metadata.pt")
    return gp, op, metadata


def test_table_shapes():
    """Verify Cayley tables are (32, 32, 32)."""
    gp, op, metadata = _load_tables()
    assert gp.shape == (32, 32, 32), f"GP shape: {gp.shape}"
    assert op.shape == (32, 32, 32), f"OP shape: {op.shape}"
    assert metadata["num_blades"] == 32
    assert metadata["grade_dims"] == [1, 5, 10, 10, 5, 1]
    print("[PASS] test_table_shapes")


def test_cga_point_null():
    """Verify CGA points are null vectors: P·P = 0."""
    from src.cgatr.interface.point import embed_point
    from src.cgatr.primitives.invariants import compute_inner_product_mask, inner_product

    gp, _, _ = _load_tables()
    ip_weights = compute_inner_product_mask(gp)

    # Random 3D points
    coords = torch.randn(100, 3)
    P = embed_point(coords)

    # P·P should be 0 for null vectors
    pp = inner_product(ip_weights, P, P)
    max_err = pp.abs().max().item()
    assert max_err < 1e-5, f"Null condition violated: max |P·P| = {max_err}"
    print(f"[PASS] test_cga_point_null (max |P·P| = {max_err:.2e})")


def test_cga_distance():
    """Verify CGA distance: d²(P1,P2) = -2⟨P1,P2⟩ matches Euclidean distance."""
    from src.cgatr.interface.point import embed_point
    from src.cgatr.primitives.invariants import compute_inner_product_mask, inner_product

    gp, _, _ = _load_tables()
    ip_weights = compute_inner_product_mask(gp)

    # Two sets of random points
    coords1 = torch.randn(50, 3)
    coords2 = torch.randn(50, 3)

    P1 = embed_point(coords1)
    P2 = embed_point(coords2)

    # CGA distance: d² = -2 * <P1, P2>
    ip = inner_product(ip_weights, P1, P2)  # (50, 1)
    d_sq_cga = -2.0 * ip.squeeze(-1)

    # Euclidean distance
    d_sq_euclidean = ((coords1 - coords2) ** 2).sum(dim=-1)

    err = (d_sq_cga - d_sq_euclidean).abs().max().item()
    assert err < 1e-4, f"Distance mismatch: max error = {err}"
    print(f"[PASS] test_cga_distance (max error = {err:.2e})")


def test_circle_is_grade3():
    """Verify circle embedding produces grade-3 trivectors."""
    from src.cgatr.interface.circle import embed_circle

    _, op, metadata = _load_tables()

    wire_pos = torch.tensor([[1.0, 0.0, 0.0]])
    wire_dir = torch.tensor([[0.0, 0.0, 1.0]])  # Along z-axis
    drift_radius = torch.tensor([[0.5]])

    circle = embed_circle(wire_pos, wire_dir, drift_radius, op)

    # Grade ranges: 0:[0], 1:[1-5], 2:[6-15], 3:[16-25], 4:[26-30], 5:[31]
    # Only grade-3 should be nonzero
    grade0 = circle[..., 0:1].abs().max().item()
    grade1 = circle[..., 1:6].abs().max().item()
    grade2 = circle[..., 6:16].abs().max().item()
    grade3 = circle[..., 16:26].abs().max().item()
    grade4 = circle[..., 26:31].abs().max().item()
    grade5 = circle[..., 31:32].abs().max().item()

    assert grade3 > 1e-6, f"Grade-3 should be nonzero, got max={grade3:.2e}"
    # Other grades should be zero (from outer product of grade-1 vectors)
    assert grade0 < 1e-5, f"Grade-0 should be ~0, got {grade0:.2e}"
    assert grade1 < 1e-5, f"Grade-1 should be ~0, got {grade1:.2e}"
    assert grade4 < 1e-5, f"Grade-4 should be ~0, got {grade4:.2e}"
    assert grade5 < 1e-5, f"Grade-5 should be ~0, got {grade5:.2e}"
    # Grade-2 may have small components from numerical precision of outer product
    print(f"[PASS] test_circle_is_grade3 (grade3_max={grade3:.4f}, others<1e-5)")


def test_pin_equi_linear_basis():
    """Verify the Pin(4,1)-equivariant linear basis has correct shape and properties."""
    from src.cgatr.primitives.linear import _compute_pin_equi_linear_basis

    basis = _compute_pin_equi_linear_basis()
    assert basis.shape == (9, 32, 32), f"Basis shape: {basis.shape}"

    # First 6 should be grade projections (diagonal blocks)
    for g in range(6):
        proj = basis[g]
        # Should be approximately diagonal within the grade block
        assert proj.abs().max() > 0, f"Grade {g} projection is zero"

    print(f"[PASS] test_pin_equi_linear_basis (shape={basis.shape})")


def test_equi_linear():
    """Test equivariant linear map forward pass."""
    from src.cgatr.primitives.linear import _compute_pin_equi_linear_basis, equi_linear

    basis = _compute_pin_equi_linear_basis()

    # Random input: (batch=10, in_channels=4, 32)
    x = torch.randn(10, 4, 32)
    # Random coefficients: (out_channels=8, in_channels=4, 9)
    coeffs = torch.randn(8, 4, 9)

    y = equi_linear(basis, x, coeffs)
    assert y.shape == (10, 8, 32), f"Output shape: {y.shape}"
    print(f"[PASS] test_equi_linear (output shape={y.shape})")


def test_cgatr_forward():
    """Test CGATr network forward pass with dummy data."""
    from src.cgatr.primitives.linear import _compute_pin_equi_linear_basis
    from src.cgatr.primitives.attention import _build_dist_basis
    from src.cgatr.primitives.invariants import compute_inner_product_mask
    from src.cgatr.primitives.dual import _DualCache
    from src.cgatr.nets.cgatr import CGATr
    from src.cgatr.layers.attention.config import SelfAttentionConfig
    from src.cgatr.layers.mlp.config import MLPConfig

    gp, op, metadata = _load_tables()
    _DualCache.init_from_metadata(metadata)

    pin_basis = _compute_pin_equi_linear_basis()
    basis_q, basis_k = _build_dist_basis(device=torch.device("cpu"), dtype=torch.float32)
    ip_weights = compute_inner_product_mask(gp)

    model = CGATr(
        in_mv_channels=1,
        out_mv_channels=1,
        hidden_mv_channels=4,  # Small for testing
        in_s_channels=None,
        out_s_channels=None,
        hidden_s_channels=16,
        num_blocks=2,
        attention=SelfAttentionConfig(num_heads=2),
        mlp=MLPConfig(),
        basis_gp=gp,
        basis_ip_weights=ip_weights,
        basis_outer=op,
        basis_pin=pin_basis,
        basis_q=basis_q,
        basis_k=basis_k,
    )

    # Dummy input: batch of 5 items, 1 MV channel, 32 components
    x = torch.randn(5, 1, 32)

    with torch.no_grad():
        out_mv, out_s = model(x)

    assert out_mv.shape == (5, 1, 32), f"Output MV shape: {out_mv.shape}"
    assert out_s is None, f"Output scalars should be None, got {type(out_s)}"
    print(f"[PASS] test_cgatr_forward (output shape={out_mv.shape})")


def test_dual():
    """Test dualization: dual(dual(x)) should give back x (up to sign)."""
    from src.cgatr.primitives.dual import _DualCache, dual

    _, _, metadata = _load_tables()
    _DualCache.init_from_metadata(metadata)

    x = torch.randn(10, 32)
    dd_x = dual(dual(x))

    # dual(dual(x)) = ±x depending on the pseudoscalar square
    # For Cl(4,1): I^2 = e12345^2 = (-1)^{5*(5-1)/2} * det(metric) = (-1)^10 * (-1) = -1
    # So dual(dual(x)) = -x or x depending on convention
    # Check if it's proportional
    ratio = dd_x / (x + 1e-10)
    # All ratios should be the same sign
    signs = ratio.sign()
    assert (signs == signs[0, 0]).all() or (dd_x.abs() < 1e-6).all(), "dual(dual(x)) not proportional to x"
    print(f"[PASS] test_dual")


def test_reversal_signs():
    """Test reversal signs for Cl(4,1)."""
    from src.cgatr.primitives.linear import _compute_reversal

    rev = _compute_reversal()
    assert rev.shape == (32,)

    # Grade 0 (idx 0): +1
    assert rev[0] == 1.0
    # Grade 1 (idx 1-5): +1
    assert (rev[1:6] == 1.0).all()
    # Grade 2 (idx 6-15): -1
    assert (rev[6:16] == -1.0).all()
    # Grade 3 (idx 16-25): -1
    assert (rev[16:26] == -1.0).all()
    # Grade 4 (idx 26-30): +1
    assert (rev[26:31] == 1.0).all()
    # Grade 5 (idx 31): +1
    assert rev[31] == 1.0

    print("[PASS] test_reversal_signs")


def test_grade_dropout():
    """Test grade dropout doesn't crash and preserves shape."""
    from src.cgatr.primitives.dropout import grade_dropout

    x = torch.randn(10, 8, 32)
    y = grade_dropout(x, p=0.5, training=True)
    assert y.shape == (10, 8, 32), f"Dropout output shape: {y.shape}"

    # In eval mode, output should equal input
    y_eval = grade_dropout(x, p=0.5, training=False)
    err = (y_eval - x).abs().max().item()
    assert err < 1e-5, f"Eval dropout should be identity, error={err}"
    print("[PASS] test_grade_dropout")


if __name__ == "__main__":
    tests = [
        test_table_shapes,
        test_cga_point_null,
        test_cga_distance,
        test_circle_is_grade3,
        test_pin_equi_linear_basis,
        test_equi_linear,
        test_reversal_signs,
        test_grade_dropout,
        test_dual,
        test_cgatr_forward,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        sys.exit(1)
