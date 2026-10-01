"""Full-production-model numerical gate for implementation-only optimizations."""

from __future__ import annotations

import argparse
import os

import torch


def _relative(got, want):
    return float(
        (
            (got - want).abs().max()
            / want.abs().max().clamp_min(1.0e-12)
        ).detach()
    )


def _set_flag(name: str, enabled: bool):
    if enabled:
        os.environ[name] = "1"
    else:
        os.environ.pop(name, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--fused", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("none", "default", "max-autotune-no-cudagraphs"),
        default="none",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    from src.eval.smoke_arms import base_args
    from src.eval.test_equivariance import make_features, rotate_features
    from src.model import CGATrParquetModel

    config = dict(
        hidden_mv_channels=16,
        hidden_s_channels=64,
        num_blocks=10,
        cga_hit_encoding="sphere_circle",
        physical_drift_geometry=True,
        separate_hit_metadata=False,
        normalize_mv_inputs=False,
        fix_cga_null=True,
        fix_wire_dir=True,
        equivariance_group="e3",
        invariant_output_head=True,
        equi_init="identity_algebra",
    )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    state = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }

    _set_flag("CGATR_SPARSE_GP", False)
    _set_flag("CGATR_FUSED_PROJECTIONS", False)
    torch.manual_seed(42)
    reference = CGATrParquetModel(base_args(**config)).cuda().train()
    reference.load_state_dict(state, strict=True)

    _set_flag("CGATR_SPARSE_GP", args.sparse)
    _set_flag("CGATR_FUSED_PROJECTIONS", args.fused)
    torch.manual_seed(42)
    candidate = CGATrParquetModel(base_args(**config)).cuda().train()
    candidate.load_state_dict(state, strict=True)
    if args.compile_mode != "none":
        candidate.compile(dynamic=True, mode=args.compile_mode)

    features_cpu = make_features(
        96, with_time=False, gen=torch.Generator().manual_seed(19)
    ).float()
    features = features_cpu.cuda()
    rotated_features = rotate_features(0.7, features_cpu).cuda()
    reference_out = reference(features, [len(features)])
    candidate_out = candidate(features, [len(features)])
    output_relative = _relative(candidate_out, reference_out)
    output_abs = float((candidate_out - reference_out).abs().max().detach())

    reference_out.square().mean().backward()
    candidate_out.square().mean().backward()
    candidate_params = dict(candidate.named_parameters())
    max_grad_relative = 0.0
    max_grad_abs = 0.0
    worst_grad = ""
    zero_grad_presence_differences = []
    for name, parameter in reference.named_parameters():
        candidate_grad = candidate_params[name].grad
        reference_grad = parameter.grad
        if candidate_grad is None and reference_grad is None:
            continue
        if candidate_grad is None or reference_grad is None:
            present_grad = (
                reference_grad if candidate_grad is None else candidate_grad
            )
            present_abs = float(present_grad.abs().max().detach())
            if present_abs > 1.0e-12:
                raise RuntimeError(
                    f"gradient presence differs for {name}, and the present "
                    f"gradient is nonzero ({present_abs:.9e})"
                )
            zero_grad_presence_differences.append(name)
            continue
        relative = _relative(candidate_grad, reference_grad)
        absolute = float(
            (candidate_grad - reference_grad).abs().max().detach()
        )
        if relative > max_grad_relative:
            max_grad_relative = relative
            worst_grad = name
        max_grad_abs = max(max_grad_abs, absolute)

    candidate.eval()
    with torch.no_grad():
        base = candidate(features, [len(features)])
        rotated = candidate(rotated_features, [len(features)])
    rotation_residual = _relative(rotated, base)

    print(
        "KERNEL_NUMERICS "
        f"sparse={int(args.sparse)} fused={int(args.fused)} "
        f"compile={args.compile_mode} "
        f"output_abs={output_abs:.9e} "
        f"output_relative={output_relative:.9e} "
        f"grad_abs={max_grad_abs:.9e} "
        f"grad_relative={max_grad_relative:.9e} "
        f"worst_grad={worst_grad} "
        f"zero_grad_presence={len(zero_grad_presence_differences)} "
        f"rotation_residual={rotation_residual:.9e}",
        flush=True,
    )
    if output_relative > 2.0e-4:
        raise RuntimeError("candidate output exceeded relative tolerance")
    if max_grad_relative > 3.0e-4:
        raise RuntimeError("candidate gradient exceeded relative tolerance")
    if rotation_residual > 2.0e-4:
        raise RuntimeError("candidate broke E(3) rotation invariance")
    if set(reference.state_dict()) != set(candidate.state_dict()):
        raise RuntimeError("candidate changed persistent checkpoint keys")


if __name__ == "__main__":
    main()
