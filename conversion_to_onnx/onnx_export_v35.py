"""ONNX export for v35 CGATr (multi-event batched).

`OnnxV35` wraps the trained `CGATrParquetModel` weights with a forward
that takes:

    features      : float32 [B, N_max, 10]
    padding_mask  : bool    [B, N_max]   (True = real hit, False = pad)

and produces

    coords_and_beta : float32 [B, N_max, embed_dim + 1]

We use the batched shape for **every** inference, even `B == 1`. The
training code uses multi-event packing (xformers `BlockDiagonalMask`,
unexportable to ONNX), so the deployment surface should mirror that:
one artifact, one code path. Single-event inference is just a
`B == 1` call with an all-True padding mask — the per-event masked
mean degenerates to a regular mean, the identity augment is a no-op,
and the NaN-zeroing never fires. Padded rows of the output carry
garbage; the caller is expected to slice with `padding_mask`.

ONNX gotchas this wrapper papers over:

  1. Attention mask. We build `[B, 1, N, N]` from `padding_mask` and
     forward it to `torch.nn.functional.scaled_dot_product_attention`
     via the `attn_mask` arg in `cgatr/primitives/attention.py`.
  2. Dual reference. The stock `CGATr._construct_dual_reference`
     averages over ALL leading dims, mixing events. We replace it
     with a **per-event masked mean** so each event's reference
     matches what running the model on that one event would give.
  3. NaN-safe padded queries. ORT's decomposed SDPA produces NaN on
     fully-padded query rows (`0/0` in softmax); torch's fused op
     hides this. We mitigate twice: OR-augment the mask with the
     identity (so every query has at least itself in scope), and
     `torch.where`-zero padded outputs after every attention/MLP
     sub-block (NaN-safe, unlike `0 *`).
  4. Tracer cleanliness. `hit_type` branching uses `torch.where`,
     einsums in `cgatr/primitives/{linear,bilinear}.py` use ellipsis
     leading dims, and `geometric_attention` reads channel counts
     from config Python ints (not tensor `.shape[-1]` SymInts).

Export uses opset 17 with `dynamic_axes` for both batch (`B`) and
max-hits (`N`); the same exported file works for any shape.

CLI (from repo root):
    PYTHONPATH=model_training python conversion_to_onnx/onnx_export_v35.py \\
        --checkpoint <ckpt.pt|.ckpt> --out <out.onnx> --embed_dim 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
MT_ROOT = REPO_ROOT / "model_training"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MT_ROOT))

from src.cgatr.nets.cgatr import CGATr
from src.cgatr.layers.attention.config import SelfAttentionConfig
from src.cgatr.layers.mlp.config import MLPConfig
from src.cgatr.interface.point import embed_point
from src.cgatr.interface.scalar import embed_scalar
from src.cgatr.interface.circle import embed_circle_ipns
from src.cgatr.primitives.linear import _compute_pin_equi_linear_basis
from src.cgatr.primitives.attention import _build_dist_basis
from src.cgatr.primitives.invariants import compute_inner_product_mask
from src.cgatr.primitives.dual import _DualCache


CGA_DIR = MT_ROOT / "cga_utils"


class OnnxV35(nn.Module):
    """Batched v35 CGATr forward, ONNX-friendly.

    See module docstring for the design notes. Weight layout mirrors
    `CGATrParquetModel` so `load_v35_weights` works for any of the
    three checkpoint formats we emit (v35 per-epoch, Lightning with
    EMA, Lightning bare).
    """

    def __init__(self, hidden_mv_channels=16, hidden_s_channels=64,
                 num_blocks=10, normalize_mv_inputs=True, embed_dim=5):
        super().__init__()
        self._normalize = normalize_mv_inputs
        self.embed_dim = embed_dim

        self.bn_pos = nn.BatchNorm1d(3, momentum=0.1)
        self.bn_wire = nn.BatchNorm1d(3, momentum=0.1)
        self.bn_drift = nn.BatchNorm1d(1, momentum=0.1)

        # Precomputed CGA Cayley tables and metadata shipped with the repo.
        gp_sparse = torch.load(str(CGA_DIR / "cga_geometric_product.pt"),
                               weights_only=False)
        self.register_buffer("basis_gp", gp_sparse.to_dense().to(torch.float32))
        op_sparse = torch.load(str(CGA_DIR / "cga_outer_product.pt"),
                               weights_only=False)
        self.register_buffer("basis_outer", op_sparse.to_dense().to(torch.float32))
        metadata = torch.load(str(CGA_DIR / "cga_metadata.pt"),
                              weights_only=False)
        _DualCache.init_from_metadata(metadata, device=torch.device("cpu"))

        pin_basis = _compute_pin_equi_linear_basis(device="cpu", dtype=torch.float32)
        basis_q, basis_k = _build_dist_basis(device="cpu", dtype=torch.float32)
        basis_ip_weights = compute_inner_product_mask(self.basis_gp, device="cpu")

        self.cgatr = CGATr(
            in_mv_channels=1, out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=None, out_s_channels=None,
            hidden_s_channels=hidden_s_channels,
            num_blocks=num_blocks,
            attention=SelfAttentionConfig(), mlp=MLPConfig(),
            basis_gp=self.basis_gp, basis_ip_weights=basis_ip_weights,
            basis_outer=self.basis_outer, basis_pin=pin_basis,
            basis_q=basis_q, basis_k=basis_k,
        )

        # Two non-equivariant linear heads on top of the 32-dim MV output:
        # clustering coords (size embed_dim) and the beta logit (size 1).
        self.clustering = nn.Linear(32, embed_dim, bias=False)
        self.beta = nn.Linear(32, 1)

    # ---- input embedding ---------------------------------------------------
    def _embed_inputs(self, features_flat: torch.Tensor) -> torch.Tensor:
        """Per-hit input embedding on a flat `[N, 10]` features tensor.

        Layout of `features_flat[i, :]`:
          [:3]   primary hit position (x, y, z)
          [3:4]  hit_type   (0.0 = vertex, 1.0 = drift chamber)
          [4:7]  DC wire reference point
          [7:8]  DC drift radius
          [8]    DC azimuthal wire angle
          [9]    DC stereo wire angle

        Vertex hits embed as conformal points; DC hits embed as the
        IPNS circle defined by (wire ref, wire direction, drift
        radius). The VTX/DC branch is a `torch.where`, not a Python
        `if`, so the trace stays shape-static.
        """
        pos = features_flat[:, :3]
        hit_type = features_flat[:, 3:4]
        is_dc = (hit_type == 1.0).to(features_flat.dtype)

        pos_normed = self.bn_pos(pos)
        wire_normed = self.bn_wire(features_flat[:, 4:7])
        drift_normed = self.bn_drift(features_flat[:, 7:8]).squeeze(-1)

        mv_vtx = embed_point(pos_normed)

        # Wire direction from (azimuth, stereo) angles.
        cos_s = torch.cos(features_flat[:, 9])
        sin_s = torch.sin(features_flat[:, 9])
        cos_a = torch.cos(features_flat[:, 8])
        sin_a = torch.sin(features_flat[:, 8])
        wire_dir = torch.stack([sin_s * cos_a, sin_s * sin_a, cos_s], dim=-1)
        wire_dir = wire_dir / (torch.norm(wire_dir, dim=-1, keepdim=True) + 1e-8)
        mv_dc = embed_circle_ipns(
            wire_normed, wire_dir, drift_normed, self.basis_outer,
        )

        mv = is_dc * mv_dc + (1.0 - is_dc) * mv_vtx
        mv = mv + embed_scalar(hit_type)  # categorical tag in the scalar slot

        if self._normalize:
            mv_norm = torch.norm(mv, dim=-1, keepdim=True).clamp(min=1e-6)
            mv = mv / mv_norm
        return mv

    # ---- CGATr stack -------------------------------------------------------
    def _run_cgatr(self, mv: torch.Tensor, attn_mask: torch.Tensor,
                   padding_mask: torch.Tensor) -> torch.Tensor:
        """Run the CGATr stack on `[B, N, 1, 32]` with per-event
        reference and NaN-safe padded-query handling.

        We reimplement `CGATr.forward` inline because both fixes need
        to slot into the residual stream between blocks; see the
        module docstring for the why.
        """
        cgatr = self.cgatr
        N = padding_mask.shape[1]

        # Boolean masks for `torch.where` (NaN-safe, unlike `*`).
        valid_mv_b = padding_mask.unsqueeze(-1).unsqueeze(-1)            # [B, N, 1, 1] bool
        valid_s_b  = padding_mask.unsqueeze(-1)                          # [B, N, 1]    bool
        valid_mv_f = valid_mv_b.to(mv.dtype)                             # [B, N, 1, 1] float
        zero_mv = torch.zeros((), device=mv.device, dtype=mv.dtype)

        # Per-event masked mean for the dual reference. Multiplying
        # by the float mask is fine here (mv is still clean: no NaN
        # can have entered yet).
        mv = torch.where(valid_mv_b, mv, zero_mv)
        n_valid = valid_mv_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        ref_sum = mv.sum(dim=1, keepdim=True)
        reference_mv = (ref_sum / n_valid).mean(dim=2, keepdim=True)

        # Identity-augmented attn_mask: every query attends to at least
        # itself, so softmax rows are never empty (no 0/0 NaN under
        # ORT's decomposed SDPA). Built with `arange + Equal` because
        # `torch.eye` exports to `EyeLike`, which ORT's default CPU EP
        # doesn't implement.
        idx = torch.arange(N, device=mv.device)
        eye = idx.unsqueeze(0) == idx.unsqueeze(1)                       # [N, N] bool
        eye = eye.unsqueeze(0).unsqueeze(0)                              # [1, 1, N, N]
        attn_mask_safe = attn_mask | eye

        def _zero_pad_mv(t: torch.Tensor) -> torch.Tensor:
            return torch.where(valid_mv_b, t, zero_mv)

        def _zero_pad_s(t: torch.Tensor) -> torch.Tensor:
            return torch.where(valid_s_b, t,
                               torch.zeros((), device=t.device, dtype=t.dtype))

        h_mv, h_s = cgatr.linear_in(mv, scalars=None)
        for block in cgatr.blocks:
            # Attention sub-block. SDPA receives `attn_mask_safe` so no
            # query has an empty row; we then `where`-zero the padded
            # query outputs so any pollution can't propagate into the
            # next block's residual stream.
            norm_mv, norm_s = block.norm(h_mv, scalars=h_s)
            attn_mv, attn_s = block.attention(
                norm_mv, scalars=norm_s,
                additional_qk_features_mv=None,
                additional_qk_features_s=None,
                attention_mask=attn_mask_safe,
            )
            attn_mv = _zero_pad_mv(attn_mv)
            if attn_s is not None:
                attn_s = _zero_pad_s(attn_s)
            h_mv = h_mv + attn_mv
            if h_s is not None and attn_s is not None:
                h_s = h_s + attn_s

            # MLP sub-block (same zeroing pattern).
            norm2_mv, norm2_s = block.norm2(h_mv, scalars=h_s)
            mlp_mv, mlp_s = block.mlp(
                norm2_mv, scalars=norm2_s, reference_mv=reference_mv,
            )
            mlp_mv = _zero_pad_mv(mlp_mv)
            if mlp_s is not None:
                mlp_s = _zero_pad_s(mlp_s)
            h_mv = h_mv + mlp_mv
            if h_s is not None and mlp_s is not None:
                h_s = h_s + mlp_s

        outputs_mv, _ = cgatr.linear_out(h_mv, scalars=h_s)
        return outputs_mv

    # ---- public forward ----------------------------------------------------
    def forward(self, features: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        """Run the batched forward.

        features     : float32 [B, N, 10]
        padding_mask : bool    [B, N]      (True = real, False = pad)
        returns      : float32 [B, N, embed_dim + 1]

        For single-event inference, pass `B == 1` and an all-True
        `padding_mask`; the result for row `[0, :n_real, :]` is bit-
        near identical to running on that one event without padding.
        """
        B, N, F = features.shape
        # Embedding is per-row; flatten so BatchNorm sees its expected
        # 2-D layout and we don't pay for a custom rank-4 BN path.
        mv_flat = self._embed_inputs(features.reshape(B * N, F))   # [B*N, 32]
        mv = mv_flat.view(B, N, 1, 32)                             # [B, N, 1, 32]

        # SDPA mask convention: True = allowed. Token i can attend to
        # token j iff both are real hits in the same event.
        pm = padding_mask
        attn_mask = (pm[:, :, None] & pm[:, None, :]).unsqueeze(1)  # [B, 1, N, N]

        out_mv = self._run_cgatr(mv, attn_mask, padding_mask)       # [B, N, 1, 32]
        out = out_mv[..., 0, :]                                     # [B, N, 32]
        return torch.cat([self.clustering(out), self.beta(out)], dim=-1)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def _extract_state_dict(raw, prefer_ema: bool = True) -> tuple[dict, str]:
    """Pull a flat `CGATrParquetModel`-shaped state_dict out of any of
    the checkpoint formats this repo emits, plus a human-readable
    label of which slot was used.

    Probed (in order, if `prefer_ema=True`):
      * Lightning ckpt with our EMA shadow: `{"ema_state_dict": {...}}`
        (already unprefixed; we wrote it via `CGATrParquetModel.state_dict()`).
      * v35 per-epoch wrapper:               `{"model_state_dict": {...}}`.
      * Lightning ckpt without EMA:          `{"state_dict": {"model.*": ...}}`
        (LightningModule auto-prefixes every param with its attribute
        name; we strip the leading `model.`).
      * Bare flat state_dict (e.g. old v34 file): used as-is.

    Returns `(state_dict, label)`.
    """
    if not isinstance(raw, dict):
        return raw, "flat state_dict"

    if prefer_ema and isinstance(raw.get("ema_state_dict"), dict):
        return raw["ema_state_dict"], "ema_state_dict"

    if isinstance(raw.get("model_state_dict"), dict):
        return raw["model_state_dict"], "model_state_dict"

    if isinstance(raw.get("state_dict"), dict):
        sd = raw["state_dict"]
        unprefixed = {
            (k[len("model."):] if k.startswith("model.") else k): v
            for k, v in sd.items()
        }
        return unprefixed, "state_dict (stripped 'model.' prefix)"

    return raw, "flat state_dict"


def load_v35_weights(model: OnnxV35, ckpt_path: str,
                     prefer_ema: bool = True) -> None:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd, label = _extract_state_dict(raw, prefer_ema=prefer_ema)
    print(f"[load] using `{label}` from {ckpt_path}")
    incompatible = model.load_state_dict(sd, strict=False)
    print(f"Missing keys:    {len(incompatible.missing_keys)}")
    print(f"Unexpected keys: {len(incompatible.unexpected_keys)}")
    for k in incompatible.missing_keys[:8]:
        print(f"  MISSING:    {k}")
    for k in incompatible.unexpected_keys[:8]:
        print(f"  UNEXPECTED: {k}")


# ---------------------------------------------------------------------------
# Synthetic inputs (used for tracing + the dynamic-shape smoke check)
# ---------------------------------------------------------------------------
def fake_event(n_vtx: int = 32, n_dc: int = 256, seed: int = 0) -> torch.Tensor:
    """A random `[n_vtx + n_dc, 10]` event with a realistic feature mix."""
    g = torch.Generator().manual_seed(seed)
    n = n_vtx + n_dc
    feats = torch.zeros(n, 10)
    feats[:n_vtx, :3] = torch.randn(n_vtx, 3, generator=g) * 200.0
    feats[:n_vtx, 3] = 0.0
    feats[n_vtx:, :3] = torch.randn(n_dc, 3, generator=g) * 1500.0
    feats[n_vtx:, 3] = 1.0
    feats[n_vtx:, 4:7] = torch.randn(n_dc, 3, generator=g) * 1500.0
    feats[n_vtx:, 7] = torch.rand(n_dc, generator=g) * 8.0
    feats[n_vtx:, 8] = torch.rand(n_dc, generator=g) * 6.28
    feats[n_vtx:, 9] = torch.rand(n_dc, generator=g) * 1.0
    perm = torch.randperm(n, generator=g)
    return feats[perm]


def build_batched_fake_input(
    batch_size: int, n_max: int, seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build `[B, N_max, 10]` features + `[B, N_max]` bool padding mask.

    Per event we draw a random number of valid hits between `n_max //
    4` and `n_max`, then zero-pad the rest. Used for tracing and the
    parity check.
    """
    g = torch.Generator().manual_seed(seed)
    features = torch.zeros(batch_size, n_max, 10)
    padding = torch.zeros(batch_size, n_max, dtype=torch.bool)
    for b in range(batch_size):
        n = int(torch.randint(n_max // 4, n_max + 1, (1,), generator=g).item())
        evt = fake_event(
            n_vtx=max(8, n // 8), n_dc=n - max(8, n // 8), seed=seed + b,
        )
        features[b, :n] = evt[:n]
        padding[b, :n] = True
    return features, padding


def wrap_single_event(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper: turn a single event `[N, 10]` into the
    batched (features, padding_mask) pair this model expects."""
    feats = features.unsqueeze(0)                                # [1, N, 10]
    pad = torch.ones(1, feats.shape[1], dtype=torch.bool,
                     device=features.device)                     # [1, N]
    return feats, pad


# ---------------------------------------------------------------------------
# Export driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Export v35 CGATr to ONNX (batched).")
    ap.add_argument("--checkpoint", default="checkpoints/cgatr_v35/cgatr_best.pt")
    ap.add_argument("--out", default="checkpoints/cgatr_v35/cgatr_v35.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument(
        "--embed_dim", type=int, default=4,
        help="Clustering-coord dim. Must match the training run that "
             "produced --checkpoint. v35-Lightning defaults to 4; "
             "legacy v35 was 5.",
    )
    ap.add_argument("--num_blocks", type=int, default=10)
    ap.add_argument("--hidden_mv_channels", type=int, default=16)
    ap.add_argument("--hidden_s_channels", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=4,
                    help="Batch size used at tracing time. The exported "
                         "graph is dynamic in `B`, so this is just a "
                         "sample shape, not a runtime constraint.")
    ap.add_argument("--n_max", type=int, default=512,
                    help="N_max used at tracing time. Dynamic in the "
                         "exported graph too.")
    ap.add_argument("--skip_export", action="store_true",
                    help="Run a torch forward only; don't write the ONNX file.")
    ap.add_argument("--skip_ort_check", action="store_true",
                    help="Skip the post-export onnxruntime parity check.")
    args = ap.parse_args()

    print(f"[ONNX spike] Building OnnxV35 (batched, embed_dim={args.embed_dim})")
    model = OnnxV35(
        num_blocks=args.num_blocks,
        hidden_mv_channels=args.hidden_mv_channels,
        hidden_s_channels=args.hidden_s_channels,
        embed_dim=args.embed_dim,
    )
    load_v35_weights(model, args.checkpoint)
    model.eval()

    x, pad = build_batched_fake_input(args.batch_size, args.n_max)
    print(f"Fake batched input: features={tuple(x.shape)} "
          f"padding_mask={tuple(pad.shape)}  "
          f"valid hits per event: {pad.sum(dim=1).tolist()}")
    with torch.no_grad():
        y_torch = model(x, pad)
    print(f"Torch forward OK. Output shape: {tuple(y_torch.shape)}, "
          f"|y|max={y_torch.abs().max().item():.4g}")

    if args.skip_export:
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ONNX spike] Exporting to {out_path} (opset={args.opset})")
    t0 = time.time()
    try:
        torch.onnx.export(
            model,
            (x, pad),
            str(out_path),
            input_names=["features", "padding_mask"],
            output_names=["coords_and_beta"],
            dynamic_axes={
                "features":        {0: "B", 1: "N"},
                "padding_mask":    {0: "B", 1: "N"},
                "coords_and_beta": {0: "B", 1: "N"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
        )
    except Exception as e:
        import traceback
        print(f"\n[ONNX spike] EXPORT FAILED:\n{type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(2)
    print(f"[ONNX spike] Export OK ({time.time() - t0:.1f}s). "
          f"File size: {out_path.stat().st_size / 1e6:.1f} MB")

    if args.skip_ort_check:
        return

    try:
        import onnxruntime as ort
    except ImportError:
        print("[ONNX spike] onnxruntime not installed — skipping parity check.")
        print("  install with: pip install onnxruntime")
        return

    print("[ONNX spike] Loading ONNX model in onnxruntime")
    sess = ort.InferenceSession(
        str(out_path), providers=["CPUExecutionProvider"],
    )

    print("[ONNX spike] Running ORT forward")
    ort_inputs = {"features": x.numpy(), "padding_mask": pad.numpy()}
    y_ort = sess.run(None, ort_inputs)[0]

    # Compare only the real-hit rows (padded rows carry garbage on both
    # sides, by design). Mask to [B, N, F] expanded.
    valid = pad.numpy()[..., None]                        # [B, N, 1]
    diff = np.abs(y_ort - y_torch.numpy()) * valid
    nan_count = int(np.isnan(y_ort).sum())
    print("[ONNX spike] Parity (real-hit rows only):")
    print(f"  max  |torch - ort|: {diff.max():.4e}")
    print(f"  mean |torch - ort|: {(diff.sum() / max(valid.sum(), 1) / y_ort.shape[-1]):.4e}")
    print(f"  ORT NaN count: {nan_count}")
    print(f"  torch range: [{y_torch.min():.3f}, {y_torch.max():.3f}]")
    print(f"  ort   range: [{float(np.nanmin(y_ort)):.3f}, "
          f"{float(np.nanmax(y_ort)):.3f}]")

    n_hits = x.shape[1]
    print(f"[ONNX spike] Timing (B={args.batch_size}, N={n_hits}):")
    with torch.no_grad():
        for _ in range(3):
            _ = model(x, pad)
        t0 = time.time()
        for _ in range(10):
            _ = model(x, pad)
        t_torch = (time.time() - t0) / 10.0

    for _ in range(3):
        _ = sess.run(None, ort_inputs)
    t0 = time.time()
    for _ in range(10):
        _ = sess.run(None, ort_inputs)
    t_ort = (time.time() - t0) / 10.0
    print(f"  torch CPU: {t_torch * 1000:.2f} ms/batch  "
          f"({t_torch * 1000 / args.batch_size:.2f} ms/event)")
    print(f"  ort   CPU: {t_ort * 1000:.2f} ms/batch  "
          f"({t_ort * 1000 / args.batch_size:.2f} ms/event)")


if __name__ == "__main__":
    main()
