"""Embedding figure for the final CIRCE model (4-D clustering embedding).

From a forward cache (forward_hits.parquet with coord_0..coord_3, beta, mc_index),
produce:
  (a) PCA scree of the 4-D embedding (variance carried per component);
  (b) t-SNE 2-D projection of representative events, colour = truth track id.

t-SNE (sklearn) is used because the local umap/numba stack is numpy-incompatible;
it answers the same "do tracks separate into compact strands" question.
Pure CPU/IO.  Usage:
  python -m src.eval.plot_embedding_final --cache <emb_all/forward_hits.parquet> --out <dir>
"""
from __future__ import annotations
import argparse, os
import numpy as np, polars as pl
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from src.eval.plotstyle import COL, TRACK_CMAP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--embed_dim", type=int, default=4)
    ap.add_argument("--beta_min", type=float, default=0.1)
    ap.add_argument("--n_events", type=int, default=4)
    ap.add_argument("--min_track_hits", type=int, default=10,
                    help="only show tracks with at least this many hits in the t-SNE")
    ap.add_argument("--hits_lo", type=int, default=900)
    ap.add_argument("--hits_hi", type=int, default=2200)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = args.embed_dim
    cc = [f"coord_{i}" for i in range(d)]

    df = pl.read_parquet(args.cache).filter(pl.col("beta") > args.beta_min)

    # ---- (a) PCA scree over all signal hits ----
    X = df.select(cc).to_numpy().astype(np.float32)
    pca = PCA(n_components=d).fit(X)
    evr = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.bar(range(1, d + 1), evr * 100, color=COL["standard"], alpha=0.85)
    ax.plot(range(1, d + 1), np.cumsum(evr) * 100, "o-", color=COL["cum"], label="cumulative")
    for i, v in enumerate(evr):
        ax.text(i + 1, v * 100 + 1, f"{v*100:.1f}%", ha="center", fontsize=8)
    ax.set_xlabel("Principal component"); ax.set_ylabel("Variance explained (%)")
    ax.set_xticks(range(1, d + 1)); ax.set_ylim(0, 105); ax.legend()
    ax.set_title(f"PCA of the {d}-D embedding ({len(X):,} signal hits)")
    fig.tight_layout(); fig.savefig(f"{args.out}/embed_final_pca_scree.pdf", bbox_inches="tight")
    fig.savefig(f"{args.out}/embed_final_pca_scree.png", dpi=150, bbox_inches="tight")
    print("PCA explained variance %:", (evr * 100).round(2).tolist())

    # ---- (b) t-SNE of a few representative-multiplicity events ----
    # pick moderate events (not the crowded tail) so the per-track structure is
    # legible, and show only substantial tracks (>= min_track_hits) so the soft
    # secondary fuzz does not swamp the picture.
    g = (df.group_by("seed", "event_id").agg(pl.len().alias("h"))
           .filter((pl.col("h") >= args.hits_lo) & (pl.col("h") <= args.hits_hi))
           .sort("h").head(args.n_events))
    if g.height < args.n_events:  # fallback if the band is too narrow
        g = (df.group_by("seed", "event_id").agg(pl.len().alias("h"))
               .sort("h").head(args.n_events))
    keys = list(zip(g["seed"].to_list(), g["event_id"].to_list()))
    ncol = 2; nrow = (len(keys) + 1) // 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for k, (s, e) in enumerate(keys):
        sub = df.filter((pl.col("seed") == s) & (pl.col("event_id") == e) & (pl.col("mc_index") > 0))
        # keep only tracks that leave at least min_track_hits hits
        vc = sub.group_by("mc_index").agg(pl.len().alias("n"))
        keep = set(vc.filter(pl.col("n") >= args.min_track_hits)["mc_index"].to_list())
        sub = sub.filter(pl.col("mc_index").is_in(list(keep)))
        Xe = sub.select(cc).to_numpy().astype(np.float32)
        y = sub["mc_index"].to_numpy()
        emb = TSNE(n_components=2, perplexity=min(30, max(5, len(Xe) // 20)),
                   init="pca", random_state=42).fit_transform(Xe)
        ax = axes[k]
        ax.scatter(emb[:, 0], emb[:, 1], s=6, c=(y % 20), cmap=TRACK_CMAP, alpha=0.9, linewidths=0)
        ax.set_title(f"{len(np.unique(y))} tracks, {len(Xe)} hits", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for k in range(len(keys), len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"CIRCE {d}-D clustering embedding, t-SNE (colour = truth track id)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{args.out}/embed_final_tsne.pdf", bbox_inches="tight")
    fig.savefig(f"{args.out}/embed_final_tsne.png", dpi=150, bbox_inches="tight")
    print("wrote", f"{args.out}/embed_final_tsne.png")


if __name__ == "__main__":
    main()
