"""Regression tests for symmetry-preserving tracking outputs."""

import pytest
import torch

from src.cgatr.primitives.linear import (
    _compute_grade_involution,
    _mirror_action_matrix,
)
from src.eval.smoke_arms import base_args
from src.eval.test_equivariance import make_features, rotate_features
from src.model import CGATrParquetModel


def _relative(got, want):
    return (
        (got - want).abs().max()
        / want.abs().max().clamp_min(1.0e-12)
    ).item()


@pytest.mark.parametrize("group", ["e3", "se3"])
def test_scalar_tracking_outputs_are_rotation_invariant(group):
    torch.manual_seed(42)
    model = CGATrParquetModel(
        base_args(
            equivariance_group=group,
            invariant_output_head=True,
            fix_cga_null=True,
            fix_wire_dir=True,
        )
    ).eval().double()
    features = make_features(
        32, with_time=False, gen=torch.Generator().manual_seed(0)
    )

    with torch.no_grad():
        outputs = model(features, [len(features)])
        moved_outputs = model(
            rotate_features(0.7, features), [len(features)]
        )

    assert outputs.shape == (len(features), 5)
    assert _relative(moved_outputs, outputs) < 2.0e-4


def test_e3_scalar_tracking_outputs_are_reflection_invariant():
    torch.manual_seed(42)
    model = CGATrParquetModel(
        base_args(
            equivariance_group="e3",
            invariant_output_head=True,
            equi_init="identity_algebra",
            fix_cga_null=True,
            fix_wire_dir=True,
        )
    ).eval().double()
    features = make_features(
        32, with_time=False, gen=torch.Generator().manual_seed(0)
    )
    mv, scalars = model.embed(features)
    mirror = _mirror_action_matrix(
        model.basis_gp,
        _compute_grade_involution(),
        vector_idx=1,
    ).to(dtype=mv.dtype)
    moved_mv = torch.einsum("ij,...j->...i", mirror, mv)

    with torch.no_grad():
        _, outputs = model._backbone_outputs(mv, scalars, [len(features)])
        _, moved_outputs = model._backbone_outputs(
            moved_mv, scalars, [len(features)]
        )

    assert outputs is not None
    assert moved_outputs is not None
    assert _relative(moved_outputs, outputs) < 2.0e-4


def test_invariant_output_head_backward_is_finite():
    torch.manual_seed(42)
    model = CGATrParquetModel(
        base_args(
            equivariance_group="e3",
            invariant_output_head=True,
            equi_init="identity_algebra",
            fix_cga_null=True,
            fix_wire_dir=True,
        )
    ).double()
    features = make_features(
        16, with_time=False, gen=torch.Generator().manual_seed(0)
    )
    outputs = model(features, [len(features)])
    outputs.square().mean().backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
