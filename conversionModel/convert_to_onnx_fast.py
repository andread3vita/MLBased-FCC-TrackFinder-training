#!/usr/bin/env python3
"""Convert a GATR_EXPERIMENT (or GATR_CIRCE_LOSS) checkpoint to a faster FP32 ONNX model.

Same contract as ``convert_to_onnx.py``: input ``hits`` float32 ``[N, 7]``
(x, y, z, planar_flag, u, v, w), output ``output`` float32 ``[N, 9]``
(4 clustering coordinates, raw beta logit, 4 zero columns kept for the
GGTFTrackFinder contract; the network has no helix head).

The network is rewritten by ``GATR_EXPERIMENT/src/models/onnx_export.py``
(folded/fused linears, outer-product + GEMM geometric product and join, export attention)
and the attention core is then rewritten on the ONNX graph:

* ``--attention blocked`` (default): standard ONNX Loop over query blocks,
  every query still attends to all keys; bounded score memory.
* ``--attention ort-mha``: ONNX Runtime ``com.microsoft.MultiHeadAttention``.
  Not a runtime-neutral file; pair it with ``--flush-denormals`` at inference.
* ``--attention dense``: plain MatMul/Softmax/MatMul.

``blocked``/``dense`` add numerical guards by default (|Q|,|K|,|V| < 1e-12
zeroed, logits floored at row_max - 30) to avoid subnormal arithmetic;
``--no-guards`` disables them. The guards and the FP32 graph are
approximations: re-check efficiency and fake rate for every checkpoint.

Usage (conversion environment, see README):

    python convert_to_onnx_fast.py -w model.ckpt -o model_fast.onnx
    python convert_to_onnx_fast.py -w model.ckpt -o model_mha.onnx --attention ort-mha
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

_MODEL_TRAINING_DIR = Path(
    os.environ.get("GGTF_MODEL_TRAINING_DIR", Path(__file__).resolve().parent.parent / "model_training")
)
_EXPERIMENT_DIR = _MODEL_TRAINING_DIR / "GATR_EXPERIMENT"
if not (_EXPERIMENT_DIR / "src" / "models" / "onnx_export.py").is_file():
    raise RuntimeError(
        f"GATR_EXPERIMENT not found at {_EXPERIMENT_DIR}; "
        "set GGTF_MODEL_TRAINING_DIR to the model_training directory"
    )
sys.path.insert(0, str(_EXPERIMENT_DIR))

from src.models import onnx_export  # noqa: E402

OPSET = 17  # ReduceMax with an ``axes`` attribute, used by the guards


def _session(onnx_path, threads, flush_denormals):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if flush_denormals:
        options.add_session_config_entry("session.set_denormal_as_zero", "1")
    return ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])


def _read_hits(csv_path):
    features = np.atleast_2d(np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32))
    if features.ndim != 2 or features.shape[1] != 7:
        raise ValueError(f"input CSV must have 7 columns, got shape {features.shape}")
    return torch.from_numpy(features)


def export(checkpoint, output, weights_source="ema", attention="blocked", block=256, guards=True):
    import onnx

    model, hparams = onnx_export.load_checkpoint(checkpoint, weights_source)
    rewritten = copy.deepcopy(model)
    counts = onnx_export.prepare_for_onnx(rewritten)
    print(f"rewritten modules: {counts}")
    dummy = torch.zeros(8, 7, dtype=torch.float32)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        raw_path = Path(directory) / "raw.onnx"
        with onnx_export.export_attention(), torch.no_grad():
            rewritten(dummy)  # fill gatr's lazily loaded basis caches before tracing
            torch.onnx.export(
                rewritten, (dummy,), str(raw_path),
                input_names=["hits"], output_names=["output"],
                dynamic_axes={"hits": {0: "N"}, "output": {0: "N"}},
                opset_version=OPSET, dynamo=False,
            )
        graph = onnx.load(str(raw_path))
    if attention == "blocked":
        rewritten_attention = onnx_export.blocked_attention(graph, block)
    elif attention == "ort-mha":
        rewritten_attention = onnx_export.runtime_attention(graph, onnx_export.num_attention_heads())
    else:
        rewritten_attention = sum(1 for n in graph.graph.node if n.op_type == "Softmax")
    print(f"attention ({attention}): {rewritten_attention} blocks")
    if guards and attention != "ort-mha":
        print(f"numerical guards: {onnx_export.stabilize_attention(graph)}")
    width = graph.graph.output[0].type.tensor_type.shape.dim[1]
    width.Clear()
    width.dim_value = model.output_dim
    onnx.checker.check_model(graph)
    onnx.save(graph, str(output))
    sidecar = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "weights_source": weights_source,
        "attention": attention,
        "block_size": block if attention == "blocked" else None,
        "guards": bool(guards and attention != "ort-mha"),
        "flush_denormals_recommended": attention == "ort-mha",
        "opset": OPSET,
        "architecture": {k: hparams.get(k) for k in
                         ("gatr_blocks", "hidden_mv_channels", "hidden_s_channels",
                          "clustering_space_dim")},
        "rewrites": counts,
    }
    output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"wrote {output} ({output.stat().st_size / 1e6:.2f} MB) and {output.with_suffix('.json').name}")
    return model


def check(model, onnx_path, hits, threads, flush_denormals, repeats):
    """Compare ONNX against the unmodified checkpoint model (fp64 and fp32 eager)."""
    session = _session(onnx_path, threads, flush_denormals)
    with torch.no_grad():
        reference64 = copy.deepcopy(model).double()(hits.double()).numpy()
        reference32 = model.float()(hits).numpy()
    feed = {"hits": hits.numpy()}
    session.run(None, feed)  # warm-up
    start = time.perf_counter()
    for _ in range(repeats):
        (result,) = session.run(None, feed)
    elapsed = (time.perf_counter() - start) / repeats
    if not np.isfinite(result).all():
        raise RuntimeError("ONNX output contains non-finite values")
    print(f"{hits.shape[0]} hits: max |ONNX - eager fp64| = {np.abs(result - reference64).max():.3e}"
          f"  (eager fp32 - fp64 for scale: {np.abs(reference32 - reference64).max():.3e})"
          f"  ONNX time {elapsed:.3f} s/event ({threads} threads,"
          f" flush denormals={'on' if flush_denormals else 'off'})")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-w", "--weightPath", required=True, help="input .ckpt file")
    parser.add_argument("-o", "--onnxPath", required=True, help="output .onnx file")
    parser.add_argument("--weights-source", choices=onnx_export.WEIGHT_SOURCES, default="ema")
    parser.add_argument("--attention", choices=onnx_export.ATTENTION_MODES, default="blocked")
    parser.add_argument("--block-size", type=int, default=256, help="query rows per Loop step (blocked)")
    parser.add_argument("--no-guards", action="store_true", help="skip the tiny-value/logit guards")
    parser.add_argument("-i", "--inputCSV", default=None,
                        help="optional event CSV (dump_model_input.py) for the final check")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--flush-denormals", action="store_true",
                        help="check with session.set_denormal_as_zero=1 (default for ort-mha)")
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per check input")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.block_size < 1 or args.threads < 1:
        raise SystemExit("--block-size and --threads must be positive")
    torch.set_num_threads(args.threads)
    model = export(args.weightPath, args.onnxPath, args.weights_source, args.attention,
                   args.block_size, not args.no_guards)
    flush = args.flush_denormals or args.attention == "ort-mha"
    torch.manual_seed(42)
    check(model, args.onnxPath, torch.randn(30, 7), args.threads, flush, args.repeats)
    if args.inputCSV:
        check(model, args.onnxPath, _read_hits(args.inputCSV), args.threads, flush, args.repeats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
