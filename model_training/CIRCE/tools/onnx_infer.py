#!/usr/bin/env python3
"""Run one measured-feature event using only NumPy and ONNX Runtime."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("features", type=Path, help="Finite float32 [hits,10] .npy event")
    parser.add_argument("--output", type=Path, default=Path("prediction.npy"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--runtime-tuned",
        action="store_true",
        help="Enable FTZ; required for the runtime-tuned export",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    features = np.load(args.features, allow_pickle=False)
    if (
        features.dtype != np.float32
        or features.ndim != 2
        or features.shape[1] != 10
        or not len(features)
        or not np.isfinite(features).all()
        or not np.isin(features[:, 3], [0, 1]).all()
    ):
        parser.error("Expected nonempty finite float32 [hits,10] features, hit_type 0 or 1")
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if args.runtime_tuned:
        options.add_session_config_entry("session.set_denormal_as_zero", "1")
    session = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])
    metadata = json.loads(session.get_modelmeta().custom_metadata_map.get("circe.export", "{}"))
    if metadata.get("variant") == "runtime-tuned" and not args.runtime_tuned:
        parser.error("This model requires --runtime-tuned (session.set_denormal_as_zero=1)")
    started = time.perf_counter()
    prediction = session.run(["prediction"], {"features": np.ascontiguousarray(features)})[0]
    elapsed = time.perf_counter() - started
    if not np.isfinite(prediction).all():
        raise RuntimeError("Non-finite prediction")
    np.save(args.output, prediction)
    print(f"{len(features)} hits; geometry + network {elapsed:.3f} s; saved {args.output}")


if __name__ == "__main__":
    main()
