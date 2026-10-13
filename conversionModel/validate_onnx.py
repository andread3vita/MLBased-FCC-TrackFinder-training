#!/usr/bin/env python3
"""Validate a converted ONNX model against the checkpoint's PyTorch inference.

By default one synthetic hit tensor (``torch.manual_seed(42)`` + ``randn``,
so the run is reproducible) is used as input; with ``-i/--inputCSV`` the
input is instead read from a CSV file produced by
``extract_model_input_csv.py`` (one hit per row, 7 float columns following
the GGTFTrackFinder feature layout: x,y,z,planar_flag,u,v,w). The input
is fed to the checkpoint's eager PyTorch inference and to
the ONNX model through ONNX Runtime; both outputs are printed together with
their shapes/dtypes, the absolute maximum difference and an allclose
verdict (rtol=1e-4, atol=1e-4).

Usage (in the conversion environment, see README):

    python validate_onnx.py -w model.ckpt -o model.onnx

The reference is the checkpoint's PyTorch inference in float64. Geometric
attention amplifies fp32 rounding with the hit count (logits up to ~1e4), so
the checkpoint's own fp32 run differs from its fp64 run by ~1e-1 on a
4k-hit event; the ONNX graph therefore computes in float64 internally and
matches the fp64 reference to ~1e-5 (erf re-implemented in float64 from
fdlibm polynomials; the residual is MKL-vs-Eigen accumulation order). The
fp32-PyTorch difference is printed for information only.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from convert_to_onnx import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch and ONNX-Runtime inference of the "
        "tracking model on one seeded random hit tensor."
    )
    parser.add_argument(
        "-w", "--weightPath", required=True, help="input .ckpt file"
    )
    parser.add_argument(
        "-o", "--onnxPath", required=True, help="converted .onnx model"
    )
    parser.add_argument(
        "-i",
        "--inputCSV",
        default=None,
        help="optional CSV with the model input (header row plus one hit per "
        "line, 7 float columns as assembled by GGTFTrackFinder; see "
        "extract_model_input_csv.py). If omitted, a seeded random 30x7 "
        "tensor is used.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import onnxruntime as ort

    model, weight_source = build_model(args.weightPath)
    model.eval()
    print(f"checkpoint weights: {weight_source}")

    if args.inputCSV is not None:
        features = np.loadtxt(args.inputCSV, delimiter=",", skiprows=1, dtype=np.float32)
        features = np.atleast_2d(features)
        if features.ndim != 2 or features.shape[1] != 7:
            print(f"input CSV must have 7 columns, got shape {features.shape}")
            return 1
        input = torch.from_numpy(features)
        print(f"input: {input.shape[0]} hits x {input.shape[1]} features from {args.inputCSV}")
    else:
        torch.manual_seed(42)
        input = torch.randn((30, 7), dtype=torch.float32)
        print("input: seeded random 30x7 tensor")

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    session = ort.InferenceSession(
        args.onnxPath, sess_options, providers=["CPUExecutionProvider"]
    )
    input_meta = session.get_inputs()[0]
    print(f"onnx input: {input_meta.name} shape {input_meta.shape} "
          f"type {input_meta.type}")

    # Reference in float64: the network is ill-conditioned in fp32 (its own
    # fp32 run is off by ~1e-1 on a 4k-hit event), and the ONNX graph
    # computes in float64 internally (convert_to_onnx.COMPUTE_DTYPE).
    with torch.no_grad():
        output_torch_fp32 = model(input).cpu()
        output_torch = model.double()(input.double()).float().cpu()

    outputs = session.run(None, {input_meta.name: input.numpy()})
    output_onnx = torch.tensor(outputs[0], device="cpu")

    print("Output Pure Python (PyTorch):", output_torch)
    print("Output ONNX Runtime:", output_onnx)

    print("\nPyTorch Output:")
    print("  Type:", type(output_torch))
    print("  Dtype:", output_torch.dtype)
    print("  Shape:", output_torch.shape)

    print("\nONNX Output:")
    print("  Type:", type(output_onnx))
    print("  Dtype:", output_onnx.dtype)
    print("  Shape:", output_onnx.shape)

    if output_torch.shape != output_onnx.shape:
        print("output shape mismatch")
        return 1

    output_torch_np = output_torch.detach().cpu().numpy()
    output_onnx_np = output_onnx.detach().cpu().numpy()
    if not np.isfinite(output_onnx_np).all():
        print("onnx output contains non-finite values")
        return 1

    max_diff = np.max(np.abs(output_torch_np - output_onnx_np))
    are_close = np.allclose(output_torch_np, output_onnx_np, rtol=1e-4, atol=1e-4)

    print("\nComparison (reference: PyTorch float64):")
    print("  Max Absolute Difference:", max_diff)
    print("  (PyTorch float32 vs ONNX, for information:",
          np.max(np.abs(output_torch_fp32.numpy() - output_onnx_np)), ")")
    print("  Are outputs close?:", are_close)
    return 0


if __name__ == "__main__":
    sys.exit(main())
