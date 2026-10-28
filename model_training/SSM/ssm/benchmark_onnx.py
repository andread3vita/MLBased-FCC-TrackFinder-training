"""ONNX Runtime CPU latency benchmark: TrackerSSM vs a quadratic-attention reference.

    python -m ssm.benchmark_onnx --ssm-onnx tracker_ssm.onnx [--n 1000 3000 6000 10000 20000]

The reference is a plain TransformerEncoder stack at comparable parameter
count, exported with the same dynamic-N contract: the mechanism by which
GATR-style models burn CPU time (N^2 attention), without GATr dependencies.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from .model import TrackerSSM


class AttnBlock(nn.Module):
    """Manual MHA block; reshapes via tensor .reshape so shapes stay dynamic at export."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.d = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.o = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, h):
        B, N, D = h.shape
        y = self.norm1(h)
        q, k, v = self.qkv(y).chunk(3, -1)
        q = q.reshape(B, N, self.h, self.d).transpose(1, 2)
        k = k.reshape(B, N, self.h, self.d).transpose(1, 2)
        v = v.reshape(B, N, self.h, self.d).transpose(1, 2)
        s = torch.matmul(q, k.transpose(-1, -2)) / self.d ** 0.5
        y = torch.matmul(torch.softmax(s, -1), v)
        y = self.o(y.transpose(1, 2).reshape(B, N, D))
        h = h + y
        return h + self.mlp(self.norm2(h))


class AttentionReference(nn.Module):
    """Quadratic-attention reference at ~matched width to TrackerSSM."""

    def __init__(self, d_model=256, n_layer=6, n_heads=8, n_features=12, embedding_dim=4):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList(AttnBlock(d_model, n_heads) for _ in range(n_layer))
        self.head = nn.Linear(d_model, embedding_dim + 1)

    def forward(self, x):
        h = self.proj(x)
        for blk in self.blocks:
            h = blk(h)
        out = self.head(h)
        return out[..., :4], out[..., 4:]


def pad_to(x: np.ndarray, multiple: int) -> np.ndarray:
    n = x.shape[1]
    n_pad = ((n + multiple - 1) // multiple) * multiple
    out = np.zeros((x.shape[0], n_pad, x.shape[2]), dtype=x.dtype)
    out[:, :n] = x
    return out


def time_session(sess, x, warmup=3, reps=None):
    reps = reps or max(3, int(3_000_000 / max(x.shape[1] ** 2, 1)) + 3)
    for _ in range(warmup):
        sess.run(None, {"features": x})
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        sess.run(None, {"features": x})
        ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts), float(np.median(ts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ssm-onnx", default=None, help="exported SSM graph (else built from scratch)")
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n", type=int, nargs="+", default=[1000, 3000, 6000, 10000, 15000, 20000])
    p.add_argument("--threads", type=int, default=0, help="ORT intra-op threads (0 = all)")
    p.add_argument("--skip-reference", action="store_true")
    args = p.parse_args()

    so = ort.SessionOptions()
    if args.threads:
        so.intra_op_num_threads = args.threads
    prov = ["CPUExecutionProvider"]

    if args.ssm_onnx:
        ssm_path, c = args.ssm_onnx, args.chunk_size
    else:
        m = TrackerSSM(d_model=args.d_model, n_layer=args.n_layer,
                       chunk_size=args.chunk_size).eval()
        torch.onnx.export(
            m, torch.zeros(1, 2 * args.chunk_size, 12), "/tmp/_bench_ssm.onnx",
            input_names=["features"], output_names=["embedding", "beta_logit"],
            dynamic_axes={"features": {1: "n"}, "embedding": {1: "n"}, "beta_logit": {1: "n"}},
            opset_version=17)
        ssm_path, c = "/tmp/_bench_ssm.onnx", args.chunk_size
    ssm = ort.InferenceSession(ssm_path, so, providers=prov)

    att = None
    if not args.skip_reference:
        ref = AttentionReference(d_model=args.d_model, n_layer=args.n_layer).eval()
        torch.onnx.export(
            ref, torch.zeros(1, 64, 12), "/tmp/_bench_att.onnx",
            input_names=["features"], output_names=["embedding", "beta_logit"],
            dynamic_axes={"features": {1: "n"}, "embedding": {1: "n"}, "beta_logit": {1: "n"}},
            opset_version=17)
        att = ort.InferenceSession("/tmp/_bench_att.onnx", so, providers=prov)

    print(f"{'N hits':>8} | {'SSM ms/ev (min/med)':>20} | {'ATTN ms/ev (min/med)':>21} | speedup")
    print("-" * 75)
    for n in args.n:
        x = np.random.randn(1, n, 12).astype(np.float32)
        s_lo, s_md = time_session(ssm, pad_to(x, c))
        if att is not None:
            a_lo, a_md = time_session(att, x)
            print(f"{n:>8} | {s_lo:>9.1f}/{s_md:<10.1f} | {a_lo:>10.1f}/{a_md:<10.1f} | {a_md/s_md:.1f}x")
        else:
            print(f"{n:>8} | {s_lo:>9.1f}/{s_md:<10.1f} | {'-':>21} |")


if __name__ == "__main__":
    main()
