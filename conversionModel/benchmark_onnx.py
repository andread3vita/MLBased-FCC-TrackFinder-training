#!/usr/bin/env python3
"""Time several tracking ONNX models on the same event(s) with ONNX Runtime.

Each model runs on identical input (one or more event CSVs from
``dump_model_input.py``, or a seeded random event) with the same session
settings; outputs are compared against the first model.

    python benchmark_onnx.py model.onnx model_fast.onnx -i event.csv --threads 4
    python benchmark_onnx.py model_fast.onnx --flush-denormals -i ev0.csv ev1.csv
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import onnxruntime as ort


def session(path, threads, flush_denormals, profile_prefix=None):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if flush_denormals:
        options.add_session_config_entry("session.set_denormal_as_zero", "1")
    if profile_prefix:
        options.enable_profiling = True
        options.profile_file_prefix = profile_prefix
    return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("models", nargs="+", help=".onnx files; the first is the reference")
    parser.add_argument("-i", "--inputCSV", nargs="*", default=[], help="event CSV file(s)")
    parser.add_argument("--random-hits", type=int, default=4000,
                        help="hits of the seeded random event used without -i")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--flush-denormals", action="store_true",
                        help="session.set_denormal_as_zero=1 for every model")
    parser.add_argument("--profile", action="store_true",
                        help="write an ONNX Runtime JSON profile per model (one timed run)")
    args = parser.parse_args()

    events = []
    for path in args.inputCSV:
        hits = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32))
        events.append((path, np.ascontiguousarray(hits)))
    if not events:
        rng = np.random.default_rng(42)
        events.append((f"random {args.random_hits} hits",
                       rng.standard_normal((args.random_hits, 7)).astype(np.float32)))

    reference = {}
    for index, path in enumerate(args.models):
        prefix = f"profile_{index}" if args.profile else None
        sess = session(path, args.threads, args.flush_denormals, prefix)
        name = sess.get_inputs()[0].name
        for label, hits in events:
            sess.run(None, {name: hits})  # warm-up
            start = time.perf_counter()
            for _ in range(1 if args.profile else args.repeats):
                (output,) = sess.run(None, {name: hits})
            elapsed = (time.perf_counter() - start) / (1 if args.profile else args.repeats)
            if index == 0:
                reference[label] = output
                diff = "reference"
            else:
                diff = f"max |diff| vs first = {np.abs(output - reference[label]).max():.3e}"
            print(f"{path}  [{label}, {hits.shape[0]} hits]  {elapsed:.3f} s/event  {diff}")
        if args.profile:
            print(f"  profile: {sess.end_profiling()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
