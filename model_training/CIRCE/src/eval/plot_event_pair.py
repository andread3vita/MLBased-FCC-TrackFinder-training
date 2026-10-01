"""Two-event truth-vs-reco display in a SINGLE image file (SN manual 7.3).

Renders two chosen events (a cleanly reconstructed one and a harder one) as one
figure: 2 rows (events) x 4 columns (3D truth, 3D reco, 2D truth, 2D reco),
reusing the drawing helpers of plot_event_display. Pure CPU.

  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python -m src.eval.plot_event_pair \
    --checkpoint <ckpt> --data_dir <dir> --clean 220 --hard 434 --seed 181
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.eval.plot_event_display import (
    load_model, forward_event, display_tracks, event_stats, draw_3d, draw_2d)
from src.eval.plotstyle import COL
from src.dataset.parquet_dataset import IDEAParquetDataset


def render_event(model, ds, idx, args):
    ev = ds[idx]
    reco, mc, sig = forward_event(model, ev, args.embed_dim, args.tbeta, args.td)
    tracks = display_tracks(mc, sig, args.min_track_hits)
    st = event_stats(reco, mc, sig, tracks)
    disp = sig & np.isin(mc, tracks)
    return ev["features"].numpy()[disp], mc[disp].astype(np.int64), reco[disp].astype(np.int64), st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--seed", type=int, default=181)
    ap.add_argument("--clean", type=int, required=True, help="dataset index of the clean event")
    ap.add_argument("--hard", type=int, required=True, help="dataset index of the harder event")
    ap.add_argument("--embed_dim", type=int, default=4)
    ap.add_argument("--num_blocks", type=int, default=10)
    ap.add_argument("--tbeta", type=float, default=0.1)
    ap.add_argument("--td", type=float, default=0.2)
    ap.add_argument("--min_track_hits", type=int, default=10)
    ap.add_argument("--out", default="/home/marko.cechovic/cgatr-paper/figures/event_display")
    args = ap.parse_args()

    model = load_model(args.checkpoint, args.num_blocks, args.embed_dim)
    ds = IDEAParquetDataset(args.data_dir, seed_range=(args.seed, args.seed + 1),
                            max_hits_per_event=0)

    events = []
    for tag, idx in (("clean", args.clean), ("hard", args.hard)):
        feats, truth, reco, st = render_event(model, ds, idx, args)
        print(f"{tag} idx {idx}: {st}")
        events.append((feats, truth, reco, st))

    # Shared square limits for the transverse panels: with per-event autoscaled
    # limits, the equal-aspect boxes get different shapes and matplotlib centres
    # them differently in their cells, misaligning the bottom row. One common
    # extent gives identical boxes AND shows both events at the same mm scale.
    def _extent(feats):
        dc = feats[:, 3] > 0.5
        xs = np.concatenate([feats[~dc, 0], feats[dc, 4] + feats[dc, 7], feats[dc, 4] - feats[dc, 7]])
        ys = np.concatenate([feats[~dc, 1], feats[dc, 5] + feats[dc, 7], feats[dc, 5] - feats[dc, 7]])
        return max(np.abs(xs).max(), np.abs(ys).max())
    lim = 1.04 * max(_extent(f) for f, _, _, _ in events)

    # 3D row on top, 2D transverse row below; the two events side by side, so
    # each event is a clean 2x2 block (matching the single-event poster look).
    fig = plt.figure(figsize=(14.5, 8.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1.0],
                          hspace=0.06, wspace=0.10)
    headers = ["Clean event", "Busier event"]
    for c, (feats, truth, reco, st) in enumerate(events):
        col = 2 * c
        ax3t = fig.add_subplot(gs[0, col], projection="3d")
        ax3r = fig.add_subplot(gs[0, col + 1], projection="3d")
        ax2t = fig.add_subplot(gs[1, col])
        ax2r = fig.add_subplot(gs[1, col + 1])
        draw_3d(ax3t, feats, truth, "Truth")
        draw_3d(ax3r, feats, reco, "CIRCE reconstruction")
        draw_2d(ax2t, feats, truth, "Truth (transverse)")
        draw_2d(ax2r, feats, reco, "CIRCE (transverse)")
        for ax in (ax2t, ax2r):
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        # per-event header spanning its two columns
        hx = 0.30 + 0.485 * c
        fig.text(hx, 0.945,
                 f"{headers[c]}: {st['n_recovered']}/{st['n_truth']} tracks recovered "
                 f"($\\langle$eff$\\rangle$ {100*st['mean_eff']:.0f}%, "
                 f"$\\langle$pur$\\rangle$ {100*st['mean_purity']:.0f}%)",
                 ha="center", va="center", fontsize=12, fontweight="bold", color=COL["ink"])
    fig.suptitle("IDEA $Z\\to q\\bar q$ events: vertex hits as dots, drift-chamber hits as drift "
                 "circles, colour = track", fontsize=12.5, y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02)
    for e in ("pdf", "png"):
        fig.savefig(f"{args.out}.{e}", dpi=200, bbox_inches="tight")
    print("wrote", f"{args.out}.png")


if __name__ == "__main__":
    main()
