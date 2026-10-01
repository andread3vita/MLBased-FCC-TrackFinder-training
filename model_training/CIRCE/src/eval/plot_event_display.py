"""Truth-vs-reco event display for the CIRCE model: 3D + 2D transverse.

Re-runs events on CPU (CGATR_FORCE_SDPA=1), clusters with the greedy beta-driven
reconstruction, scans a handful of events and picks a *legible, well-reconstructed*
one (a moderate number of primary tracks, most of them cleanly recovered), then
draws a 2x2 figure:
    top row    — 3D view (poster style): VTX hits as dots, DC hits as drift circles
    bottom row — 2D transverse (x,y) view of the same event
left column = truth particle colouring, right column = reconstructed cluster.
The geometry-native drift circle (radius = drift distance about the wire) is the
model's actual input representation. Pure CPU.

Usage:
  CUDA_VISIBLE_DEVICES="" CGATR_FORCE_SDPA=1 PYTHONPATH=. python -m src.eval.plot_event_display \
      --checkpoint <ckpt> --data_dir <dir> --seed 181 --embed_dim 4
"""
from __future__ import annotations
import argparse, os
import numpy as np, torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import src.cgatr.primitives.attention as _attn
_attn._HAS_XFORMERS = False  # force CPU-compatible SDPA path (single-event forward)

from src.model import CGATrParquetModel
from src.dataset.parquet_dataset import IDEAParquetDataset
from src.eval_sweep_v33 import get_clustering_greedy
from src.eval.forward_pass import _make_args
from src.eval.plotstyle import COL  # applies the shared theme on import

# qualitative palette for track ids (shared look with the t-SNE embedding figure)
PALETTE = np.concatenate([plt.cm.tab20.colors, plt.cm.tab20b.colors, plt.cm.tab20c.colors])
NOISE_RGBA = (0.72, 0.72, 0.72, 0.55)


def load_model(ckpt, num_blocks, embed_dim):
    m = CGATrParquetModel(_make_args(num_blocks=num_blocks, embed_dim=embed_dim,
                                     hidden_mv_channels=16, hidden_s_channels=64))
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "state_dict" in st:
        sd = st.get("ema_state_dict") or st["state_dict"]
        st = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
    elif "model_state_dict" in st:
        st = st["model_state_dict"]
    m.load_state_dict(st, strict=True); m.eval()
    return m


def _remap(labels):
    uniq = [u for u in np.unique(labels) if u >= 0]
    m = {u: i for i, u in enumerate(uniq)}
    return np.array([m.get(l, -1) for l in labels]), len(uniq)


def _colors(local_labels):
    return np.array([NOISE_RGBA if l < 0 else (*PALETTE[int(l) % len(PALETTE)][:3], 0.9)
                     for l in local_labels])


def _wire_basis(wdir):
    w = wdir / np.linalg.norm(wdir, axis=1, keepdims=True).clip(min=1e-9)
    helper = np.tile(np.array([1.0, 0.0, 0.0]), (w.shape[0], 1))
    helper[np.abs(w[:, 0]) > 0.9] = np.array([0.0, 1.0, 0.0])
    u = np.cross(w, helper); u /= np.linalg.norm(u, axis=1, keepdims=True).clip(min=1e-9)
    v = np.cross(w, u); v /= np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-9)
    return u, v


def _drift_circles_3d(wire_xyz, drift, azim, stereo, n_seg=24):
    if len(wire_xyz) == 0:
        return np.zeros((0, n_seg + 1, 3), np.float32)
    # Detector convention: a stereo wire tilts azimuthally, not radially. This
    # draws true geometry, so it is not gated behind --fix_wire_dir.
    wdir = np.stack([np.sin(stereo) * np.sin(azim), -np.sin(stereo) * np.cos(azim),
                     np.cos(stereo)], -1)
    u, v = _wire_basis(wdir)
    t = np.linspace(0, 2 * np.pi, n_seg + 1)
    ct, st = np.cos(t)[None, :, None], np.sin(t)[None, :, None]
    return (wire_xyz[:, None, :] + drift[:, None, None]
            * (u[:, None, :] * ct + v[:, None, :] * st))


def draw_3d(ax, feats, labels, title):
    is_dc = feats[:, 3] > 0.5
    local, n = _remap(labels)
    # DC drift circles
    dc = feats[is_dc]
    if len(dc):
        circ = _drift_circles_3d(dc[:, 4:7], dc[:, 7], dc[:, 8], dc[:, 9])
        ax.add_collection3d(Line3DCollection(circ, colors=_colors(local[is_dc]), linewidths=0.5))
    # VTX dots
    vtx = feats[~is_dc]
    if len(vtx):
        ax.scatter(vtx[:, 0], vtx[:, 1], vtx[:, 2], c=_colors(local[~is_dc]),
                   s=18, depthshade=False, edgecolors="black", linewidths=0.25)
    pts = feats[is_dc][:, 4:7] if is_dc.any() else feats[:, :3]
    if is_dc.any() and (~is_dc).any():
        pts = np.vstack([pts, feats[~is_dc][:, :3]])
    mins, maxs = pts.min(0), pts.max(0); pad = 0.02 * (maxs - mins).max()
    ax.set_xlim(mins[0] - pad, maxs[0] + pad); ax.set_ylim(mins[1] - pad, maxs[1] + pad)
    ax.set_zlim(mins[2] - pad, maxs[2] + pad)
    ax.set_xlabel("X [mm]", labelpad=-8); ax.set_ylabel("Y [mm]", labelpad=-8)
    ax.set_zlabel("Z [mm]", labelpad=-8)
    ax.tick_params(labelsize=6, pad=-2); ax.grid(False)
    ax.set_title(title, fontsize=11)
    ax.view_init(elev=20, azim=35)


def draw_2d(ax, feats, labels, title):
    is_dc = feats[:, 3] > 0.5
    local, _ = _remap(labels)
    cols = _colors(local)
    # DC drift circles (transverse projection ~ circle of radius=drift about wire xy)
    for i in np.where(is_dc)[0]:
        ax.add_patch(Circle((feats[i, 4], feats[i, 5]), max(feats[i, 7], 1.0),
                            fill=False, ec=cols[i], lw=0.35, alpha=0.8))
    vtx = ~is_dc
    ax.scatter(feats[vtx, 0], feats[vtx, 1], c=cols[vtx], s=8, linewidths=0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(title, fontsize=11)


@torch.no_grad()
def forward_event(model, ev, embed_dim, tbeta, td):
    out = model(ev["features"], [ev["n_hits"]])
    coords = out[:, :embed_dim].numpy().astype(np.float32)
    beta = torch.sigmoid(out[:, embed_dim]).numpy().astype(np.float32)
    mc = ev["mc_index"].numpy(); sec = ev["is_secondary"].numpy().astype(bool)
    sig = (~sec) & (mc > 0)
    reco_full = np.full(len(mc), -1, np.int64)
    reco_full[sig] = get_clustering_greedy(beta[sig], coords[sig], tbeta=tbeta, td=td)
    return reco_full, mc, sig


def display_tracks(mc, sig, min_hits):
    """Primary tracks that leave at least `min_hits` hits (the substantial,
    ~reconstructable tracks worth drawing; soft 1-2-hit primaries are dropped)."""
    prim = mc[sig]
    vals, counts = np.unique(prim, return_counts=True)
    return vals[counts >= min_hits]


def event_stats(reco, mc, sig, tracks):
    """Per-track reconstruction stats over the given (substantial) truth tracks:
    for each, match to the reco cluster sharing the most hits and record its hit
    efficiency (shared/|track|) and purity (shared/|cluster|). Returns a dict with
    the counts and the mean efficiency/purity, plus the count cleanly recovered."""
    effs, purs, recovered = [], [], 0
    for t in tracks:
        t_hits = np.where(sig & (mc == t))[0]
        labs = reco[t_hits]; labs = labs[labs >= 0]
        if len(labs) == 0:
            effs.append(0.0); purs.append(0.0); continue
        c = np.bincount(labs).argmax()
        shared = (labs == c).sum()
        eff = shared / len(t_hits); pur = shared / (reco == c).sum()
        effs.append(eff); purs.append(pur)
        if pur >= 0.75 and eff >= 0.5:
            recovered += 1
    n = max(len(tracks), 1)
    return {"n_truth": len(tracks), "n_recovered": recovered,
            "mean_eff": float(np.mean(effs)) if effs else 0.0,
            "mean_purity": float(np.mean(purs)) if purs else 0.0,
            "frac_recovered": recovered / n}


def quality(reco, mc, sig, tracks):
    return event_stats(reco, mc, sig, tracks)["frac_recovered"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--seed", type=int, default=181)
    ap.add_argument("--embed_dim", type=int, default=4)
    ap.add_argument("--num_blocks", type=int, default=10)
    ap.add_argument("--tbeta", type=float, default=0.1)
    ap.add_argument("--td", type=float, default=0.2)
    ap.add_argument("--min_tracks", type=int, default=15)
    ap.add_argument("--max_tracks", type=int, default=35)
    ap.add_argument("--min_track_hits", type=int, default=10,
                    help="only draw/count primary tracks with at least this many hits")
    ap.add_argument("--scan_events", type=int, default=18)
    ap.add_argument("--event_index", type=int, default=-1,
                    help="force a specific dataset index instead of scanning")
    ap.add_argument("--out", default="/home/marko.cechovic/cgatr-paper/figures/event_display")
    args = ap.parse_args()

    model = load_model(args.checkpoint, args.num_blocks, args.embed_dim)
    ds = IDEAParquetDataset(args.data_dir, seed_range=(args.seed, args.seed + 1),
                            max_hits_per_event=0)

    # ---- pass 1 (no forward): list events with a legible primary-track count ----
    cands = []
    for idx in range(len(ds)):
        ev = ds[idx]
        if ev is None:
            continue
        mc = ev["mc_index"].numpy(); sec = ev["is_secondary"].numpy().astype(bool)
        sig = (~sec) & (mc > 0)
        n_disp = len(display_tracks(mc, sig, args.min_track_hits))
        n_sig = int(sig.sum())
        if args.event_index >= 0:
            if idx == args.event_index:
                cands = [(n_sig, idx)]; break
            continue
        if args.min_tracks <= n_disp <= args.max_tracks:
            cands.append((n_sig, idx))
    # smallest signal-hit events first: faster forwards and cleaner displays
    cands.sort()
    cands = cands[:args.scan_events]
    if not cands:
        raise SystemExit("no event with a substantial-track count in the requested window")

    # ---- pass 2: forward + cluster the candidates, keep the best-reconstructed ----
    best = None
    for _, idx in cands:
        ev = ds[idx]
        reco, mc, sig = forward_event(model, ev, args.embed_dim, args.tbeta, args.td)
        tracks = display_tracks(mc, sig, args.min_track_hits)
        q = quality(reco, mc, sig, tracks)
        print(f"  event idx {idx}: {len(tracks)} substantial tracks, {int(sig.sum())} signal hits, "
              f"quality={q:.2f}", flush=True)
        score = (q, len(tracks))
        if best is None or score > best[0]:
            best = (score, idx, ev, reco, mc, sig)

    (_, idx, ev, reco, mc, sig) = best
    feats = ev["features"].numpy()
    tracks = display_tracks(mc, sig, args.min_track_hits)
    # draw only the substantial primary tracks, for a clean truth-vs-reco display
    disp = sig & np.isin(mc, tracks)
    feats_s = feats[disp]
    truth_s = mc[disp].astype(np.int64)
    reco_s = reco[disp].astype(np.int64)
    st = event_stats(reco, mc, sig, tracks)
    print(f"chosen event idx {idx}: {st}")

    from src.eval.plotstyle import COL
    fig = plt.figure(figsize=(11, 9.8))
    ax3t = fig.add_subplot(2, 2, 1, projection="3d")
    ax3r = fig.add_subplot(2, 2, 2, projection="3d")
    ax2t = fig.add_subplot(2, 2, 3)
    ax2r = fig.add_subplot(2, 2, 4)
    draw_3d(ax3t, feats_s, truth_s, f"Truth: {st['n_truth']} tracks")
    draw_3d(ax3r, feats_s, reco_s,
            f"CIRCE: {st['n_recovered']}/{st['n_truth']} tracks recovered")
    draw_2d(ax2t, feats_s, truth_s, "Truth (transverse)")
    draw_2d(ax2r, feats_s, reco_s, "CIRCE reconstruction (transverse)")
    fig.suptitle("IDEA $Z\\to q\\bar q$ event: vertex hits as dots, drift-chamber hits as "
                 "drift circles; colour = track", fontsize=12, y=0.99)
    # stats strip along the top (plain unicode; no LaTeX escapes since usetex is off)
    stats = (f"{st['n_truth']} truth tracks    ·    "
             f"{st['n_recovered']} reconstructed (purity ≥ 75%, efficiency ≥ 50%)"
             f"    ·    ⟨efficiency⟩ {100*st['mean_eff']:.0f}%"
             f"    ·    ⟨purity⟩ {100*st['mean_purity']:.0f}%")
    fig.text(0.5, 0.945, stats, ha="center", va="center", fontsize=10.5,
             color=COL["ink"],
             bbox=dict(boxstyle="round,pad=0.5", fc="#EEF3FF", ec="#C9D6F0", lw=1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    for e in ("pdf", "png"):
        fig.savefig(f"{args.out}.{e}", dpi=200)
    print("wrote", f"{args.out}.png")


if __name__ == "__main__":
    main()
