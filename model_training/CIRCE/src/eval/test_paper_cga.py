"""Regression tests for the paper-faithful FCC C-GATr input path."""

import torch

from src.cgatr.interface.circle import embed_circle_ipns
from src.cgatr.interface.line import embed_line
from src.cgatr.interface.plane import embed_plane
from src.cgatr.interface.point import embed_point
from src.cgatr.interface.sphere import embed_sphere
from src.cgatr.primitives.bilinear import outer_product
from src.cgatr.primitives.invariants import compute_inner_product_mask
from src.eval.smoke_arms import base_args
from src.eval.test_equivariance import (
    make_features,
    rotate_features,
    translate_features,
)
from src.model import CGATrParquetModel


def _args(**overrides):
    return base_args(
        cga_hit_encoding="sphere_circle",
        physical_drift_geometry=True,
        separate_hit_metadata=False,
        normalize_mv_inputs=False,
        fix_cga_null=True,
        fix_wire_dir=True,
        equivariance_group="e3",
        invariant_output_head=True,
        equi_init="identity_algebra",
        **overrides,
    )


def _relative(got, want):
    return (
        (got - want).abs().max()
        / want.abs().max().clamp_min(1.0e-12)
    ).item()


def _wire_direction(features):
    azimuth, stereo = features[:, 8], features[:, 9]
    direction = torch.stack(
        [
            torch.sin(stereo) * torch.sin(azimuth),
            -torch.sin(stereo) * torch.cos(azimuth),
            torch.cos(stereo),
        ],
        dim=-1,
    )
    return direction / direction.norm(dim=-1, keepdim=True)


def _transform_features(features, matrix):
    """Apply an arbitrary orthogonal transform to raw positions and wires."""
    transformed = features.clone()
    transformed[:, :3] = features[:, :3] @ matrix.T
    transformed[:, 4:7] = features[:, 4:7] @ matrix.T
    is_dc = features[:, 3] == 1
    direction = _wire_direction(features[is_dc]) @ matrix.T
    transformed[is_dc, 8] = torch.atan2(direction[:, 0], -direction[:, 1])
    transformed[is_dc, 9] = torch.acos(direction[:, 2].clamp(-1.0, 1.0))
    return transformed


def test_conformal_line_is_incident_and_point_gauge_independent():
    model = CGATrParquetModel(_args(num_blocks=1)).eval().double()
    position = torch.tensor(
        [[0.3, -0.4, 0.7], [-0.8, 0.2, -0.1]], dtype=torch.float64
    )
    direction = torch.tensor(
        [[0.2, 0.9, 0.3], [-0.4, 0.1, 0.8]], dtype=torch.float64
    )
    direction = direction / direction.norm(dim=-1, keepdim=True)
    line = embed_line(position, direction, model.basis_outer)
    shifted_line = embed_line(
        position + 2.7 * direction, direction, model.basis_outer
    )
    torch.testing.assert_close(line, shifted_line, atol=2e-13, rtol=2e-13)

    point_on_line = embed_point(position - 1.3 * direction)
    incidence = outer_product(model.basis_outer, point_on_line, line)
    assert incidence.abs().max().item() < 2.0e-13

    assert line[..., :16].abs().max().item() < 2.0e-13
    assert line[..., 16:26].abs().max().item() > 1.0e-6
    assert line[..., 26:].abs().max().item() < 2.0e-13


def test_paper_cga_model_boundary_and_full_rigid_motion_invariance():
    torch.manual_seed(42)
    model = CGATrParquetModel(_args(num_blocks=2)).eval().double()
    features = make_features(
        32, with_time=False, gen=torch.Generator().manual_seed(0)
    )
    mv, scalars = model.embed(features)
    is_vtx = features[:, 3] == 0
    is_dc = ~is_vtx

    torch.testing.assert_close(
        mv[is_vtx, 0],
        embed_point(features[is_vtx, :3] / model.pos_scale),
        atol=1e-12,
        rtol=1e-12,
    )
    wire_position = features[is_dc, 4:7] / model.pos_scale
    drift_radius = features[is_dc, 7] / model.pos_scale
    wire_direction = _wire_direction(features[is_dc])
    sphere = embed_sphere(
        wire_position, drift_radius, fix_null=True
    )
    sphere_with_hit_type = sphere.clone()
    sphere_with_hit_type[:, 0] = 1.0
    circle = embed_circle_ipns(
        wire_position,
        wire_direction,
        drift_radius,
        model.basis_outer,
        fix_null=True,
    )
    torch.testing.assert_close(
        mv[is_dc, 0],
        sphere_with_hit_type,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        mv[is_dc, 1],
        circle,
        # model.embed uses the same normalization with a 1e-8 denominator
        # guard; this is its bounded effect on an already unit wire vector.
        atol=1e-7,
        rtol=1e-7,
    )
    # A point at the measured radius in the measured normal plane satisfies
    # both constraints whose wedge is the explicit circle.
    axis = torch.zeros_like(wire_direction)
    axis[:, 0] = 1.0
    radial = torch.cross(wire_direction, axis, dim=-1)
    use_y = radial.norm(dim=-1) < 1.0e-6
    axis[use_y] = torch.tensor(
        [0.0, 1.0, 0.0], dtype=axis.dtype, device=axis.device
    )
    radial = torch.cross(wire_direction, axis, dim=-1)
    radial = radial / radial.norm(dim=-1, keepdim=True)
    tangency_point = embed_point(
        wire_position + drift_radius[:, None] * radial
    )
    plane = embed_plane(
        wire_direction, wire_position, fix_null=True
    )
    ip_weights = compute_inner_product_mask(model.basis_gp)
    sphere_incidence = (
        ip_weights * tangency_point * sphere
    ).sum(dim=-1).abs().max()
    plane_incidence = (
        ip_weights * tangency_point * plane
    ).sum(dim=-1).abs().max()
    assert sphere_incidence.item() < 2.0e-12
    assert plane_incidence.item() < 2.0e-12
    assert torch.count_nonzero(mv[is_vtx, 1]) == 0
    # Radius occurs exactly once, in the sphere/circle geometry. Hit type uses
    # the grade-0 scalar blade, matching GGTF; there are no auxiliary scalars.
    assert scalars is None

    with torch.no_grad():
        outputs = model(features, [len(features)])
        rotated = model(rotate_features(0.7, features), [len(features)])
        translated = model(
            translate_features((310.0, -170.0, 440.0), features),
            [len(features)],
        )
        mirror = torch.diag(
            torch.tensor([-1.0, 1.0, 1.0], dtype=features.dtype)
        )
        reflected = model(
            _transform_features(features, mirror), [len(features)]
        )
    assert _relative(rotated, outputs) < 2.0e-10
    # The production basis is solved in float64 and stored in float32 before
    # the model is promoted to double here, so the finite translation carries
    # the expected one-time float32 basis-rounding residual.
    assert _relative(translated, outputs) < 2.0e-7
    assert _relative(reflected, outputs) < 2.0e-10
