#!/usr/bin/env python3
"""Package a tracking checkpoint as a LibTorch-loadable TorchScript module."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import torch

from event_features import load_real_event_features

from export_tracking_onnx import (
    ExactTrainingExportModel,
    configure_export_einsum,
    make_training_args,
    select_state,
    strip_lightning_prefix,
    torch_load,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--weights-source",
        "--weights",
        dest="weights_source",
        choices=("ema", "raw"),
        default="ema",
        help="checkpoint weights to package (default: EMA validation weights)",
    )
    parser.add_argument("--num-hits", type=int, default=16)
    parser.add_argument(
        "--verify-parquet",
        type=Path,
        help="also require bit-for-bit equality on this real-event parquet file",
    )
    parser.add_argument("--event-index", type=int, default=0)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.num_hits < 1:
        raise ValueError("--num-hits must be positive")
    if args.event_index < 0:
        raise ValueError("--event-index must be non-negative")
    if args.output.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("output must have a .pt or .pth suffix")

    checkpoint = torch_load(args.checkpoint)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("The checkpoint is not a dictionary")
    selected, source = select_state(checkpoint, args.weights_source)
    state = strip_lightning_prefix(selected)

    configure_export_einsum()
    model = ExactTrainingExportModel(make_training_args(checkpoint))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch. Missing keys: {missing}; "
            f"unexpected keys: {unexpected}"
        )
    model.eval()

    example = torch.zeros(
        args.num_hits, model.input_feature_count, dtype=torch.float32
    )
    with torch.inference_mode():
        # Materialize the immutable GATr algebra and join kernels before tracing.
        expected = model(example)
        traced = torch.jit.trace(
            model,
            example,
            check_trace=False,
            strict=False,
        )
        actual = traced(example)
        # Exercise a different token count to ensure the trace did not freeze
        # the dynamic event dimension.
        dynamic_example = torch.zeros(
            args.num_hits + 1, model.input_feature_count, dtype=torch.float32
        )
        dynamic_output = traced(dynamic_example)
        if args.verify_parquet is not None:
            real_features = load_real_event_features(
                args.verify_parquet, args.event_index
            )
            real_expected = model(real_features)
            real_actual = traced(real_features)
        else:
            real_expected = real_actual = None

    if not torch.equal(expected, actual):
        difference = torch.max(torch.abs(expected - actual)).item()
        raise RuntimeError(
            "TorchScript tracing changed the example output; "
            f"maximum absolute difference is {difference}"
        )
    expected_dynamic_shape = (args.num_hits + 1, model.embedding_dim + 5)
    if tuple(dynamic_output.shape) != expected_dynamic_shape:
        raise RuntimeError(
            f"Dynamic TorchScript check returned {tuple(dynamic_output.shape)}, "
            f"expected {expected_dynamic_shape}"
        )
    if real_expected is not None and not torch.equal(real_expected, real_actual):
        difference = torch.max(torch.abs(real_expected - real_actual)).item()
        raise RuntimeError(
            "TorchScript differs from the checkpoint-derived model on the "
            f"real event; maximum absolute difference is {difference}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(args.output))
    reloaded = torch.jit.load(str(args.output), map_location="cpu")
    reloaded.eval()
    with torch.inference_mode():
        reloaded_output = reloaded(example)
    if not torch.equal(expected, reloaded_output):
        difference = torch.max(torch.abs(expected - reloaded_output)).item()
        raise RuntimeError(
            "Reloading the TorchScript module changed its output; "
            f"maximum absolute difference is {difference}"
        )

    print(f"Packaged exact {source} checkpoint weights as {args.output}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Input: features[num_hits, {model.input_feature_count}]")
    print("Model input: geometry only; detector features are unsupported")
    print(f"Output: predictions[num_hits, {model.embedding_dim + 5}]")
    print("TorchScript eager/trace/reload checks are bit-for-bit identical")
    if real_expected is not None:
        print(
            "Real-event TorchScript check passed bit-for-bit: "
            f"event={args.event_index}, hits={real_expected.shape[0]}"
        )


if __name__ == "__main__":
    main()
