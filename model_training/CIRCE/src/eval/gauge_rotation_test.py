"""Does the drift encoding depend on a choice nature never made?

A drift hit is a wire line and a radius. The set of possible track positions is
a cylinder around the wire, so it is symmetric under rotation about the wire:
nothing physical distinguishes one point of the drift circle from another.

GGTF's parquet nevertheless picks two of them, left and right, at w +- r * x',
where x' = normalise([1, 0, -d_x/d_z]) is built from wire geometry alone. That
is a gauge choice. Rotating x' about the wire by any angle names the same hit.

So rotate it, and see who notices:

  - the conformal arm embeds the circle itself, from (wire, wire_dir, radius),
    and never reads x'. Its input is bit-identical, so it is exactly invariant.
  - the projective 'ggtf' arm is built from the two points and nothing else, so
    its input moves, and whatever it learned about x' moves with it.

The projective arm's degradation is partly ordinary distribution shift, which
is the point rather than a caveat: a representation that has to learn a
convention has spent capacity on something that carries no physics, and stays
brittle to it.

Usage:
    python -m src.eval.gauge_rotation_test --events 300
"""

import argparse
import time

import numpy as np
import torch

from src.eval.forward_pass import (_load_state_dict_tolerating_derived_buffers,
                                   _make_args)
from src.model import CGATrParquetModel, get_clustering_np
from src.dataset.parquet_dataset import IDEAParquetDataset

DATA = "/home/marko.cechovic/cgatr-data/data-final/parquet"
RUNS = "/home/marko.cechovic/cgatr-runs/parity"
EMBED_DIM = 4  # matches launch_ggtf_parity.sh

ARMS = {
    "projective_parity": dict(algebra="projective", pga_hit_encoding="ggtf",
                              hidden_mv_channels=26),
    "conformal_parity": dict(algebra="conformal", pga_hit_encoding="line",
                             hidden_mv_channels=16, fix_cga_null=True),
}


def wire_dir_true(azim, stereo):
    """The detector convention; see src/eval/audit_drift_encoding.py."""
    return torch.stack([torch.sin(stereo) * torch.sin(azim),
                        -torch.sin(stereo) * torch.cos(azim),
                        torch.cos(stereo)], -1)


def rotate_about(v, axis, theta):
    """Rodrigues rotation of v about a unit axis."""
    c, s = np.cos(theta), np.sin(theta)
    return (v * c
            + torch.cross(axis, v, dim=-1) * s
            + axis * (axis * v).sum(-1, keepdim=True) * (1.0 - c))


def regauge(features, theta, use_time=False):
    """Re-pick the two tangency points at a different angle around the wire.

    The wire, the radius and therefore the drift circle are untouched; only
    which pair of its points the parquet happens to name changes.
    """
    f = features.clone()
    is_dc = f[:, 3] == 1
    if not bool(is_dc.any()):
        return f
    off = 11 if use_time else 10
    axis = wire_dir_true(f[is_dc, 8], f[is_dc, 9])
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
    u = f[is_dc, off:off + 3]
    u = rotate_about(u, axis, theta)
    f[is_dc, off:off + 3] = u / (u.norm(dim=-1, keepdim=True) + 1e-8)
    return f


def build(arm, device):
    cfg = dict(num_blocks=10, embed_dim=EMBED_DIM, hidden_s_channels=64,
               legacy_equivariance=False, use_time=False)
    cfg.update(ARMS[arm])
    model = CGATrParquetModel(_make_args(**cfg)).to(device)
    ck = torch.load(f"{RUNS}/{arm}/cgatr_best.ckpt", map_location="cpu",
                    weights_only=False)
    state = ck.get("ema_state_dict") or ck["state_dict"]
    state = {k[len("model."):] if k.startswith("model.") else k: v
             for k, v in state.items()}
    _load_state_dict_tolerating_derived_buffers(model, state)
    model.eval()
    return model


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", type=int, default=300)
    p.add_argument("--seeds", default="191-192")
    p.add_argument("--tbeta", type=float, default=0.6)
    p.add_argument("--td", type=float, default=0.2)
    p.add_argument("--max_hits", type=int, default=1200,
                   help="Skip larger events. Attention is quadratic and this "
                        "runs on CPU while the GPUs are busy training.")
    p.add_argument("--threads", type=int, default=2,
                   help="Keep low: the training dataloaders already "
                        "oversubscribe this box.")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    lo, hi = (int(x) for x in args.seeds.split("-"))
    dataset = IDEAParquetDataset(DATA, seed_range=(lo, hi + 1),
                                 with_drift_dir=True)
    device = torch.device(args.device)
    thetas = [0.0, np.pi / 4, np.pi / 2, np.pi]

    print(f"{len(dataset)} events available, using up to {args.events} "
          f"with at most {args.max_hits} hits")
    print(f"device {device}, clustering at tbeta={args.tbeta} td={args.td}\n")

    picked = []
    for idx in range(len(dataset)):
        if len(picked) >= args.events:
            break
        ev = dataset[idx]
        if ev is None or not 8 <= ev["n_hits"] <= args.max_hits:
            continue
        if bool((ev["features"][:, 3] == 1).any()):
            picked.append(ev)
    print(f"selected {len(picked)} events, "
          f"{sum(e['n_hits'] for e in picked):,} hits total\n")

    # The cheap half, and the exact one: what the rotation does to the inputs
    # before any learned weights are involved.
    print("Input under re-gauging (embedding only, no backbone):")
    for arm in ARMS:
        model = build(arm, device)
        worst = 0.0
        for ev in picked:
            f0 = ev["features"]
            mv0, _ = model.embed(f0)
            for t in thetas[1:]:
                mvt, _ = model.embed(regauge(f0, t))
                worst = max(worst, float((mvt - mv0).abs().max()))
        verdict = "exactly invariant" if worst == 0.0 else "moves"
        print(f"    {arm:20s} max|d embedding| over all rotations = {worst:.3e}  ({verdict})")
    print()

    for arm in ARMS:
        model = build(arm, device)
        t0 = time.time()
        d_in = {t: [] for t in thetas}
        d_out = {t: [] for t in thetas}
        d_lab = {t: [] for t in thetas}
        n = 0
        for ev in picked:
            f0 = ev["features"]
            n += 1
            seq = [ev["n_hits"]]
            o0 = model(f0.to(device), seq)
            c0 = o0[:, :EMBED_DIM].cpu().numpy()
            b0 = torch.sigmoid(o0[:, EMBED_DIM]).cpu().numpy()
            l0 = get_clustering_np(b0, c0, tbeta=args.tbeta, td=args.td)
            scale = np.linalg.norm(c0) + 1e-12
            for t in thetas:
                ft = regauge(f0, t).to(device)
                ot = model(ft, seq)
                ct = ot[:, :EMBED_DIM].cpu().numpy()
                bt = torch.sigmoid(ot[:, EMBED_DIM]).cpu().numpy()
                lt = get_clustering_np(bt, ct, tbeta=args.tbeta, td=args.td)
                d_out[t].append(float(np.linalg.norm(ct - c0) / scale))
                d_lab[t].append(float((lt != l0).mean()))

        print(f"--- {arm}  ({n} events, {time.time() - t0:.0f}s)")
        print(f"    {'rotation':>10s}  {'rel d output':>13s}  {'hits reclustered':>17s}")
        for t in thetas:
            print(f"    {np.degrees(t):>9.0f}d  {np.mean(d_out[t]):>13.2e}"
                  f"  {np.mean(d_lab[t]) * 100:>16.2f}%")
        print()


if __name__ == "__main__":
    main()
