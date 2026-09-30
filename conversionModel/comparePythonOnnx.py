#!/usr/bin/env python3
"""Compare the current Python and ONNX GATR models on one real parquet event."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent / "model_training" / "GATR"
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(TRAINING))

from export_tracking_onnx import (  # noqa: E402
    ExactTrainingExportModel,
    configure_export_einsum,
    make_training_args,
    select_state,
    strip_lightning_prefix,
    torch_load,
)
from event_features import load_real_event_features  # noqa: E402


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("onnx", type=Path)
    p.add_argument("parquet", type=Path)
    p.add_argument("--event-index", type=int, default=0)
    p.add_argument(
        "--weights-source",
        "--weights",
        dest="weights_source",
        choices=("ema", "raw"),
        default="ema",
        help="checkpoint weights used by both Python and ONNX comparison",
    )
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument(
        "--gaudi-dump-prefix",
        type=Path,
        help=(
            "compare against <prefix>_event<index>_input.bin and "
            "<prefix>_event<index>_output.bin dumped by GGTFTrackFinder"
        ),
    )
    return p.parse_args()


def read_gaudi_tensor(path):
    with path.open("rb") as stream:
        header = stream.read(16)
        if len(header) != 16:
            raise RuntimeError(f"Invalid tensor dump header: {path}")
        rows, columns = struct.unpack("=QQ", header)
        values = np.frombuffer(stream.read(), dtype=np.float32)
    expected = rows * columns
    if values.size != expected:
        raise RuntimeError(
            f"Invalid tensor dump payload: {path} contains {values.size} "
            f"floats, expected {expected}"
        )
    return values.reshape(rows, columns)


def print_comparison(label, reference, candidate, rtol, atol):
    if reference.shape != candidate.shape:
        raise RuntimeError(
            f"{label} shape mismatch: reference={reference.shape}, "
            f"candidate={candidate.shape}"
        )
    difference = np.abs(reference - candidate)
    worst = np.unravel_index(np.argmax(difference), difference.shape)
    print(f"{label}:")
    print(f"  exactly equal: {bool(np.array_equal(reference, candidate))}")
    print(f"  close (rtol={rtol:g}, atol={atol:g}): "
          f"{bool(np.allclose(reference, candidate, rtol=rtol, atol=atol))}")
    print(f"  maximum absolute difference: {float(difference[worst]):.8g}")
    print(f"  mean absolute difference:    {float(difference.mean()):.8g}")
    print(
        f"  largest difference at row={worst[0]}, column={worst[1]}: "
        f"reference={reference[worst]:.8g}, candidate={candidate[worst]:.8g}"
    )
    return bool(np.allclose(reference, candidate, rtol=rtol, atol=atol))


def load_python_model(checkpoint, args, state):
    # This is the same vendored GATr v1.4.2 core used by the exporter and training;
    # it avoids Lightning/DGL graph plumbing for the tensor-level comparison.
    configure_export_einsum()
    model = ExactTrainingExportModel(args)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Python checkpoint mismatch; missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    return model


def main():
    args = arguments()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)
    if not args.parquet.is_file():
        raise FileNotFoundError(args.parquet)
    if args.event_index < 0:
        raise ValueError("--event-index must be non-negative")

    checkpoint = torch_load(args.checkpoint)
    selected, source = select_state(checkpoint, args.weights_source)
    state = strip_lightning_prefix(selected)
    training_args = make_training_args(checkpoint)
    features = load_real_event_features(args.parquet, args.event_index)
    python_model = load_python_model(checkpoint, training_args, state)

    with torch.inference_mode():
        python_output = python_model(features).cpu().numpy()

    session = ort.InferenceSession(
        str(args.onnx), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: features.numpy()})[0]

    if python_output.shape != onnx_output.shape:
        raise RuntimeError(
            f"Shape mismatch: Python={python_output.shape}, ONNX={onnx_output.shape}"
        )

    difference = np.abs(python_output - onnx_output)
    max_difference = float(difference.max(initial=0.0))
    mean_difference = float(difference.mean())
    close = bool(np.allclose(
        python_output, onnx_output, rtol=args.rtol, atol=args.atol
    ))

    print(f"Checkpoint weights: {source}")
    print(f"Parquet event: {args.parquet} (index={args.event_index})")
    print(f"Number of hits: {features.shape[0]}")
    print(f"Python output shape: {python_output.shape}")
    print(f"ONNX output shape:  {onnx_output.shape}")
    print(f"Python ONNX Runtime version: {ort.__version__}")
    print(f"Maximum absolute difference: {max_difference:.8g}")
    print(f"Mean absolute difference:    {mean_difference:.8g}")
    print(f"Outputs close: {close}")

    gaudi_comparisons_close = True
    if args.gaudi_dump_prefix is not None:
        prefix = str(args.gaudi_dump_prefix)
        gaudi_input = read_gaudi_tensor(
            Path(f"{prefix}_event{args.event_index}_input.bin")
        )
        gaudi_output = read_gaudi_tensor(
            Path(f"{prefix}_event{args.event_index}_output.bin")
        )
        input_close = print_comparison(
            "Parquet features versus Gaudi inference input",
            features.numpy(),
            gaudi_input,
            args.rtol,
            args.atol,
        )
        checkpoint_gaudi_close = print_comparison(
            "Python checkpoint versus Gaudi inference output",
            python_output,
            gaudi_output,
            args.rtol,
            args.atol,
        )
        # This comparison is diagnostic only. Different ONNX Runtime releases
        # can choose different float32 reduction kernels even when both are
        # correct. The requested end-to-end check is checkpoint versus Gaudi,
        # with an exact parquet-versus-Gaudi input check guarding preprocessing.
        print_comparison(
            "Python ONNX Runtime versus Gaudi inference output",
            onnx_output,
            gaudi_output,
            args.rtol,
            args.atol,
        )
        gaudi_comparisons_close = input_close and checkpoint_gaudi_close

    if not close:
        worst = np.unravel_index(np.argmax(difference), difference.shape)
        print(
            f"Largest difference at hit={worst[0]}, output={worst[1]}: "
            f"python={python_output[worst]:.8g}, onnx={onnx_output[worst]:.8g}"
        )
        if args.gaudi_dump_prefix is None:
            raise SystemExit(1)
    if not gaudi_comparisons_close:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
