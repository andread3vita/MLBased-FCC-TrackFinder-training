"""Export TrackerSSM to ONNX (dynamic padded-sequence axis) + ORT parity check.

    python -m ssm.export_onnx --ckpt last.ckpt --out tracker_ssm.onnx
    python -m ssm.export_onnx --random-init --out tracker_ssm.onnx   # no ckpt needed

Graph contract (single event):
  input  : features (1, N_pad, 12) float32, N_pad multiple of chunk_size
  outputs: embedding (1, N_pad, 4) float32, beta_logit (1, N_pad, 1) float32
Slice to the first N_real rows downstream.
"""

from __future__ import annotations

import argparse

import onnx
import onnxruntime as ort
import torch

from .model import TrackerSSM
import numpy as np


def build_model(args) -> TrackerSSM:
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu")
        saved = ck.get("args", {})
        m = TrackerSSM(
            d_model=saved.get("d_model", args.d_model),
            n_layer=saved.get("n_layer", args.n_layer),
            expand=saved.get("expand", args.expand),
            n_heads=saved.get("n_heads", args.n_heads),
            d_state=saved.get("d_state", args.d_state),
            chunk_size=saved.get("chunk_size", args.chunk_size),
            embedding_dim=saved.get("embedding_dim", args.embedding_dim),
            bidirectional=saved.get("bidirectional", args.bidirectional),
        )
        m.load_state_dict(ck["model"])
    else:
        m = TrackerSSM(
            d_model=args.d_model, n_layer=args.n_layer, expand=args.expand,
            n_heads=args.n_heads, d_state=args.d_state, chunk_size=args.chunk_size,
            embedding_dim=args.embedding_dim, bidirectional=args.bidirectional,
        )
    return m.eval()


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--random-init", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--expand", type=int, default=1)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-state", type=int, default=64)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--embedding-dim", type=int, default=4)
    p.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


def main():
    args = args_parser()
    model = build_model(args)
    c = model.chunk_size
    dummy = torch.zeros(1, 2 * c, 12)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["features"],
        output_names=["embedding", "beta_logit"],
        dynamic_axes={
            "features": {1: "n_hits_padded"},
            "embedding": {1: "n_hits_padded"},
            "beta_logit": {1: "n_hits_padded"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(args.out))
    print(f"exported -> {args.out}")

    # ORT parity across several dynamic lengths
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    for k in (1, 2, 3, 5):
        x = torch.randn(1, k * c, 12)
        with torch.no_grad():
            e_ref, b_ref = model(x)
        e_on, b_on = sess.run(None, {"features": x.numpy()})
        de = np.abs(e_on - e_ref.numpy()).max()
        db = np.abs(b_on - b_ref.numpy()).max()
        status = "OK" if de < 1e-4 and db < 1e-4 else "MISMATCH"
        print(f"N={k*c:5d}  max|d_emb|={de:.2e}  max|d_beta|={db:.2e}  [{status}]")


if __name__ == "__main__":
    main()
