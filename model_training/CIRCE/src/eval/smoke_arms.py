"""CPU smoke test for the ablation arms.

Builds each arm, runs one real event through it, and checks the output shape and
finiteness. Runs on CPU by design so it cannot disturb a training job.

    CUDA_VISIBLE_DEVICES= python -m src.eval.smoke_arms
"""

import argparse
import sys

import torch

from src.dataset.parquet_dataset import IDEAParquetDataset, collate_idea_events
from src.model import CGATrParquetModel

DATA = "/home/marko.cechovic/cgatr-data/data-final/parquet"


def base_args(**over):
    a = argparse.Namespace(
        num_blocks=2,
        hidden_mv_channels=8,
        hidden_s_channels=16,
        embed_dim=4,
        normalize_mv_inputs=True,
        grad_checkpoint=False,
        use_time=False,
        algebra="conformal",
        legacy_equivariance=False,
        equivariance_group="se3",
        invariant_output_head=False,
        pga_hit_encoding="line",
        two_channel_dc=False,
        cga_hit_encoding="circle",
        physical_drift_geometry=False,
        separate_hit_metadata=False,
        equi_init="default",
        fix_cga_null=False,
        fix_wire_dir=False,
    )
    for k, v in over.items():
        setattr(a, k, v)
    return a


def run_arm(name, args, seed):
    model = CGATrParquetModel(args)
    ds = IDEAParquetDataset(DATA, seed_range=(seed, seed + 1),
                            max_hits_per_event=400,
                            with_time=args.use_time,
                            with_drift_dir=model.needs_drift_dir)
    if len(ds) == 0:
        raise SystemExit(f"no events found under {DATA}/seed_{seed}")

    batch = collate_idea_events([ds[0], ds[1]])
    feats, seq_lens = batch["features"], batch["seq_lens"]

    expect_cols = (10 + (1 if args.use_time else 0)
                   + (3 if model.needs_drift_dir else 0))
    assert feats.shape[1] == expect_cols, \
        f"{name}: features have {feats.shape[1]} cols, expected {expect_cols}"

    n_par = sum(p.numel() for p in model.parameters())

    model.eval()
    with torch.no_grad():
        out = model(feats, seq_lens)

    assert out.shape == (feats.shape[0], args.embed_dim + 1), \
        f"{name}: output shape {tuple(out.shape)}"
    assert torch.isfinite(out).all(), f"{name}: non-finite output"

    tcol = ""
    if args.use_time:
        t = feats[:, 10]
        tcol = f"  time[min/med/max] {t.min():+.2f}/{t.median():+.2f}/{t.max():+.2f}"
    print(f"  [ok] {name:<22} hits={feats.shape[0]:<5} params={n_par:>9,} "
          f"out={tuple(out.shape)}{tcol}")
    return n_par


def production_args(**over):
    """The width the real runs use, for parameter matching."""
    cfg = dict(num_blocks=10, hidden_mv_channels=16, hidden_s_channels=64)
    cfg.update(over)
    return base_args(**cfg)


def report_params():
    """Parameter counts at production width, and the projective width that
    matches the conformal arm.

    The projective algebra carries 16 blades against 32 and admits 16 SE(3)
    equivariant
    linear maps against 40, so equal channel counts would hand the conformal arm
    far more capacity and confound the comparison.
    """
    ref = sum(p.numel() for p in CGATrParquetModel(production_args()).parameters())
    print(f"conformal (hidden_mv_channels=16): {ref:,} parameters")

    best = None
    for mv in range(20, 33, 2):
        n = sum(p.numel() for p in
                CGATrParquetModel(production_args(algebra="projective",
                                                  hidden_mv_channels=mv)).parameters())
        d = abs(n - ref) / ref
        print(f"  projective mv={mv:<3} {n:>12,}  ({n / ref:.2f}x conformal)")
        if best is None or d < best[1]:
            best = (mv, d, n)
    print(f"\nclosest match: MV_PROJECTIVE={best[0]} "
          f"({best[2]:,} parameters, {best[1] * 100:.1f}% from conformal)")


def check_join(seed=1):
    """The join must not be identically zero.

    It is half the hidden width in the projective arm, and a degenerate join
    would silently halve that arm's capacity -- the exact way a self-built
    baseline ends up unfairly weak. The conformal code dropped its join branch
    because the reference multivector zeroed it, so this is a real failure mode.
    """
    model = CGATrParquetModel(base_args(algebra="projective")).eval()
    bilinear = model.cgatr.blocks[0].mlp.layers[0]
    assert bilinear.join is not None, "projective arm built without a join"

    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(32, model.args.hidden_mv_channels, model.num_blades,
                    generator=gen)
    with torch.no_grad():
        left, _ = bilinear.linear_left(x)
        right, _ = bilinear.linear_right(x)
        gp = bilinear.geometric_product(left, right)
        jn = bilinear.join(left, right)
    ratio = jn.abs().mean().item() / max(gp.abs().mean().item(), 1e-12)
    ok = jn.abs().max().item() > 1e-8
    print(f"  [{'ok' if ok else 'FAIL'}] join branch is "
          f"{'non-degenerate' if ok else 'IDENTICALLY ZERO'} "
          f"(mean |join| / mean |gp| = {ratio:.3f})")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--report-params", action="store_true")
    a = p.parse_args()

    if a.report_params:
        report_params()
        return

    if torch.cuda.is_available():
        print("[warn] CUDA visible; rerun with CUDA_VISIBLE_DEVICES= to stay off "
              "the training GPUs", file=sys.stderr)

    print("Smoke-testing ablation arms on CPU")
    run_arm("conformal (as-is)", base_args(legacy_equivariance=True), a.seed)
    run_arm("conformal (corrected)", base_args(), a.seed)
    run_arm("conformal (E3 paper)", base_args(equivariance_group="e3"), a.seed)
    run_arm(
        "conformal (E3 invariant output)",
        base_args(equivariance_group="e3", invariant_output_head=True),
        a.seed,
    )
    run_arm("conformal + time", base_args(use_time=True), a.seed)
    run_arm("projective (iP-GATr)", base_args(algebra="projective"), a.seed)
    run_arm("projective (GGTF enc)",
            base_args(algebra="projective", pga_hit_encoding="ggtf"), a.seed)
    # v2 ingredients
    run_arm("conformal null-fixed", base_args(fix_cga_null=True), a.seed)
    run_arm("conformal 2-channel", base_args(two_channel_dc=True), a.seed)
    run_arm("conformal identity-init",
            base_args(equi_init="identity_algebra"), a.seed)
    run_arm("conformal v2 stack",
            base_args(fix_cga_null=True, two_channel_dc=True, use_time=True,
                      equi_init="identity_algebra"), a.seed)
    run_arm("conformal physical constraints",
            base_args(fix_cga_null=True, fix_wire_dir=True,
                      cga_hit_encoding="sphere_plane",
                      physical_drift_geometry=True,
                      separate_hit_metadata=True), a.seed)
    run_arm(
        "conformal paper circle",
        base_args(
            fix_cga_null=True,
            fix_wire_dir=True,
            cga_hit_encoding="sphere_circle",
            physical_drift_geometry=True,
            separate_hit_metadata=False,
            normalize_mv_inputs=False,
            equivariance_group="e3",
            invariant_output_head=True,
            equi_init="identity_algebra",
        ),
        a.seed,
    )
    run_arm("projective physical point+line",
            base_args(algebra="projective", fix_wire_dir=True,
                      pga_hit_encoding="point_line",
                      physical_drift_geometry=True,
                      separate_hit_metadata=True), a.seed)
    if not check_join(a.seed):
        raise SystemExit("join branch is degenerate")
    print("all arms passed")


if __name__ == "__main__":
    main()
