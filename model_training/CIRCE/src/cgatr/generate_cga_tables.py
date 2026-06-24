"""Generate Cayley tables for Conformal Geometric Algebra Cl(4,1).

Basis vectors: e1, e2, e3 (square to +1), e+ (square to +1), e- (square to -1).
32 basis blades, grades 0-5 with dimensions (1, 5, 10, 10, 5, 1).

Usage:
    pip install clifford
    python -m src.cgatr.generate_cga_tables
"""

import os
from pathlib import Path

import torch


def generate_cga_tables(output_dir: str = None):
    """Generate and save CGA Cayley tables.

    Parameters
    ----------
    output_dir : str or None
        Directory to save tables. Defaults to model_training/cga_utils/.
    """
    try:
        from clifford import Cl
        import numpy as np
    except ImportError:
        raise ImportError("Install clifford: pip install clifford")

    # Create Cl(4,1): 4 positive, 1 negative
    layout, blades = Cl(4, 1)

    num_blades = 2**5  # 32
    assert len(layout.bladeTupList) == num_blades

    # Build blade ordering by grade, using clifford's internal indices
    # layout.bladeTupList gives tuples like (), (1,), (1,2), etc.
    # layout._basis_blade_order.index_to_bitmap maps internal index to bitmap
    blade_tuples_by_grade = []  # Our reordering: sorted by grade
    blade_grades = []
    grade_ranges = {}

    # Collect blades grouped by grade
    idx = 0
    for grade in range(6):
        start = idx
        for internal_idx, blade_tuple in enumerate(layout.bladeTupList):
            if len(blade_tuple) == grade:
                blade_tuples_by_grade.append((internal_idx, blade_tuple))
                blade_grades.append(grade)
                idx += 1
        grade_ranges[grade] = (start, idx)

    assert len(blade_tuples_by_grade) == num_blades, (
        f"Expected 32 blades, got {len(blade_tuples_by_grade)}"
    )

    # Verify grade dimensions: (1, 5, 10, 10, 5, 1)
    for g in range(6):
        s, e = grade_ranges[g]
        expected = [1, 5, 10, 10, 5, 1][g]
        assert e - s == expected, f"Grade {g}: expected {expected} blades, got {e - s}"

    # Create basis multivectors using the value array directly
    # layout.bladeTupList[internal_idx] -> blade tuple
    # We need a mapping from our index to clifford's internal index
    our_to_internal = [item[0] for item in blade_tuples_by_grade]
    blade_names = [item[1] for item in blade_tuples_by_grade]

    def make_basis_blade(our_idx):
        """Create a multivector with 1.0 at the given blade (our ordering)."""
        mv = layout.MultiVector()
        internal_idx = our_to_internal[our_idx]
        mv.value[internal_idx] = 1.0
        return mv

    def extract_coeffs(mv):
        """Extract coefficients in our blade ordering."""
        coeffs = np.zeros(num_blades)
        for our_idx in range(num_blades):
            internal_idx = our_to_internal[our_idx]
            coeffs[our_idx] = mv.value[internal_idx]
        return coeffs

    # Compute geometric product table: gp[i, j, k] = coefficient of blade_i in blade_j * blade_k
    print("Computing geometric product table...")
    gp_table = torch.zeros(num_blades, num_blades, num_blades, dtype=torch.float32)

    for j in range(num_blades):
        bv_j = make_basis_blade(j)
        for k in range(num_blades):
            bv_k = make_basis_blade(k)
            product = bv_j * bv_k  # Geometric product
            coeffs = extract_coeffs(product)
            for i in range(num_blades):
                if abs(coeffs[i]) > 1e-10:
                    gp_table[i, j, k] = coeffs[i]

    # Compute outer product table
    print("Computing outer product table...")
    op_table = torch.zeros(num_blades, num_blades, num_blades, dtype=torch.float32)

    for j in range(num_blades):
        bv_j = make_basis_blade(j)
        for k in range(num_blades):
            bv_k = make_basis_blade(k)
            product = bv_j ^ bv_k  # Outer (wedge) product
            coeffs = extract_coeffs(product)
            for i in range(num_blades):
                if abs(coeffs[i]) > 1e-10:
                    op_table[i, j, k] = coeffs[i]

    # Compute dualization: dual(blade_j) = blade_j * I^{-1}
    print("Computing dualization...")
    # Pseudoscalar I = e12345 is the last blade in our ordering (grade 5)
    I_mv = make_basis_blade(num_blades - 1)  # grade-5 blade
    I_inv = I_mv.inv()

    dual_perm = []
    dual_signs = torch.zeros(num_blades, dtype=torch.float32)

    for j in range(num_blades):
        bv_j = make_basis_blade(j)
        d = bv_j * I_inv
        coeffs = extract_coeffs(d)

        # Find which blade this maps to
        found = False
        for i in range(num_blades):
            if abs(coeffs[i]) > 1e-10:
                dual_perm.append(i)
                dual_signs[j] = coeffs[i]
                found = True
                break
        assert found, f"Could not find dual of blade {j} ({blade_names[j]})"

    # Compute reversal signs: ~(e_{i1...ik}) = (-1)^{k(k-1)/2} e_{i1...ik}
    reversal_signs = torch.zeros(num_blades, dtype=torch.float32)
    for i, grade in enumerate(blade_grades):
        reversal_signs[i] = (-1.0) ** (grade * (grade - 1) // 2)

    # Compute grade involution signs: hat(e_{i1...ik}) = (-1)^k e_{i1...ik}
    grade_involution_signs = torch.zeros(num_blades, dtype=torch.float32)
    for i, grade in enumerate(blade_grades):
        grade_involution_signs[i] = (-1.0) ** grade

    # Save everything
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent.parent / "cga_utils")

    os.makedirs(output_dir, exist_ok=True)

    # Save as sparse tensors for consistency with PGA format
    gp_sparse = gp_table.to_sparse()
    op_sparse = op_table.to_sparse()

    torch.save(gp_sparse, os.path.join(output_dir, "cga_geometric_product.pt"))
    torch.save(op_sparse, os.path.join(output_dir, "cga_outer_product.pt"))

    # Save metadata
    metadata = {
        "blade_names": blade_names,
        "blade_grades": blade_grades,
        "grade_ranges": grade_ranges,
        "dual_permutation": dual_perm,
        "dual_signs": dual_signs,
        "reversal_signs": reversal_signs,
        "grade_involution_signs": grade_involution_signs,
        "signature": (4, 1),
        "num_blades": num_blades,
        "grade_dims": [1, 5, 10, 10, 5, 1],
    }
    torch.save(metadata, os.path.join(output_dir, "cga_metadata.pt"))

    print(f"Saved CGA tables to {output_dir}")
    print(f"  Geometric product: {gp_table.shape} ({gp_table.nonzero().shape[0]} nonzero)")
    print(f"  Outer product: {op_table.shape} ({op_table.nonzero().shape[0]} nonzero)")
    print(f"  Blade ordering: {blade_names}")
    print(f"  Grade ranges: {grade_ranges}")
    print(f"  Dual permutation: {dual_perm}")
    print(f"  Dual signs: {dual_signs.tolist()}")
    print(f"  Reversal signs: {reversal_signs.tolist()}")

    return gp_table, op_table, metadata


if __name__ == "__main__":
    generate_cga_tables()
