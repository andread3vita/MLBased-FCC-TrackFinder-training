"""Pin the preprocessing used by the corrected algebra comparison.

This test works at the model boundary, after every scale and metadata choice.
It therefore catches the /1000-versus-/5 mismatch that primitive-only tests
cannot see.

    CUDA_VISIBLE_DEVICES= python -m src.eval.test_physical_encodings
"""

import torch

from src.cgatr.interface.plane import embed_plane
from src.cgatr.interface.point import embed_point
from src.cgatr.interface.sphere import embed_sphere
from src.cgatr.primitives.invariants import compute_inner_product_mask
from src.dataset.parquet_dataset import IDEAParquetDataset
from src.eval.smoke_arms import DATA, production_args
from src.model import CGATrParquetModel


def _normalise(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _wire_direction(dc):
    a, s = dc[:, 8], dc[:, 9]
    out = torch.stack(
        [torch.sin(s) * torch.sin(a),
         -torch.sin(s) * torch.cos(a),
         torch.cos(s)],
        dim=-1,
    )
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _ip(weights, a, b):
    return (weights * a * b).sum(dim=-1)


def main():
    ds = IDEAParquetDataset(
        DATA, seed_range=(1, 2), max_hits_per_event=500, with_drift_dir=True)
    features = ds[0]["features"]
    dc = features[features[:, 3] == 1][:128]
    assert len(dc) > 0

    cga_args = production_args(
        fix_cga_null=True,
        fix_wire_dir=True,
        cga_hit_encoding="sphere_plane",
        physical_drift_geometry=True,
        separate_hit_metadata=True,
    )
    cga = CGATrParquetModel(cga_args).eval()
    cga_mv, cga_s = cga.embed(dc)

    w = dc[:, 4:7] / cga.pos_scale
    r_geometry = dc[:, 7] / cga.pos_scale
    wire_dir = _wire_direction(dc)
    sphere = embed_sphere(w, r_geometry, fix_null=True)
    plane = embed_plane(wire_dir, w, fix_null=True)
    expected_cga = torch.stack([_normalise(sphere), _normalise(plane)], dim=1)
    torch.testing.assert_close(cga_mv, expected_cga, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        cga_s, torch.stack([dc[:, 7] / 5.0, dc[:, 3]], dim=-1),
        atol=1e-7, rtol=1e-7)

    # The exact model-scaled sphere and plane must contain the physical
    # left/right tangency points. This is the relation the legacy /5 geometry
    # violated by a factor of 200 in radius.
    u = dc[:, 10:13]
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    left = w - r_geometry[:, None] * u
    point = embed_point(left)
    ipw = compute_inner_product_mask(cga.basis_gp)
    sphere_resid = _ip(ipw, point, sphere).abs().max().item()
    plane_resid = _ip(ipw, point, plane).abs().max().item()
    assert sphere_resid < 2e-6, sphere_resid
    assert plane_resid < 2e-6, plane_resid

    pga_args = production_args(
        algebra="projective",
        hidden_mv_channels=26,
        fix_wire_dir=True,
        pga_hit_encoding="point_line",
        physical_drift_geometry=True,
        separate_hit_metadata=True,
    )
    pga = CGATrParquetModel(pga_args).eval()
    pga_mv, pga_s = pga.embed(dc)

    homog = torch.cat([w, torch.ones_like(w[:, :1])], dim=-1)
    point_on_wire = homog @ pga.point_matrix.T
    moment = torch.cross(w, wire_dir, dim=-1)
    plucker = torch.cat([wire_dir, moment], dim=-1)
    line = plucker @ pga.line_matrix.T
    expected_pga = torch.stack(
        [_normalise(point_on_wire), _normalise(line)], dim=1)
    torch.testing.assert_close(pga_mv, expected_pga, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(pga_s, cga_s, atol=1e-7, rtol=1e-7)

    n_cga = sum(p.numel() for p in cga.parameters())
    n_pga = sum(p.numel() for p in pga.parameters())
    mismatch = abs(n_cga - n_pga) / n_cga
    assert mismatch < 0.05, (n_cga, n_pga, mismatch)

    # A real model-level backward pass catches channel-count and scalar-routing
    # errors that embedding equality alone cannot see.
    for name, model in (("CGA", cga), ("PGA", pga)):
        model.train()
        output = model(dc[:64], [64])
        assert torch.isfinite(output).all(), name
        output.square().mean().backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert any(g is not None for g in grads), name
        assert all(torch.isfinite(g).all() for g in grads if g is not None), name
        model.zero_grad(set_to_none=True)

    print(
        "physical encodings and backward passes passed | "
        f"sphere incidence={sphere_resid:.2e} "
        f"plane incidence={plane_resid:.2e} | "
        f"params CGA={n_cga:,} PGA={n_pga:,} ({mismatch * 100:.1f}% apart)"
    )


if __name__ == "__main__":
    main()
