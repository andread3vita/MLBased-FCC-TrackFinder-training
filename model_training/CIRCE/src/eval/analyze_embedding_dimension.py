"""Audit whether a learned 4-D clustering embedding uses its fourth dimension.

Fit PCA on validation outputs only, apply the locked transform to validation and
test caches, and write 3-D caches that can be passed through the normal
operating-point evaluator. Optional UMAP diagnostics are event-local: embeddings
from different events may overlap arbitrarily, so a global UMAP would not have a
meaningful track-neighbour interpretation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def _coordinates(df: pl.DataFrame, dim: int = 4) -> np.ndarray:
    return df.select([f"coord_{i}" for i in range(dim)]).to_numpy().astype(
        np.float32, copy=False
    )


def _project_cache(
    source: str,
    destination: Path,
    pca: PCA,
) -> int:
    df = pl.read_parquet(source)
    projected = pca.transform(_coordinates(df))[:, :3].astype(np.float32)
    df = df.with_columns(
        [pl.Series(f"coord_{i}", projected[:, i]) for i in range(3)]
    ).drop("coord_3")
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(destination, compression="zstd")
    return df.height


def _event_keys(df: pl.DataFrame, n_events: int) -> list[tuple[int, int]]:
    counts = (
        df.group_by(["seed", "event_id"])
        .agg(pl.len().alias("n"))
        .filter((pl.col("n") >= 500) & (pl.col("n") <= 3000))
        .sort("n")
    )
    if counts.height == 0:
        counts = (
            df.group_by(["seed", "event_id"])
            .agg(pl.len().alias("n"))
            .sort("n")
        )
    positions = np.linspace(0, max(counts.height - 1, 0), n_events).round().astype(int)
    positions = np.unique(positions)
    return [
        (int(counts["seed"][int(i)]), int(counts["event_id"][int(i)]))
        for i in positions
    ]


def _neighbour_metrics(original: np.ndarray, projected: np.ndarray, truth: np.ndarray):
    k = min(10, len(original) - 1)
    if k < 1:
        return {"n": len(original), "knn_overlap": 1.0, "same_track_4d": 1.0,
                "same_track_pca3": 1.0}
    nn4 = NearestNeighbors(n_neighbors=k + 1).fit(original)
    nn3 = NearestNeighbors(n_neighbors=k + 1).fit(projected)
    i4 = nn4.kneighbors(return_distance=False)[:, 1:]
    i3 = nn3.kneighbors(return_distance=False)[:, 1:]
    overlap = np.mean(
        [len(set(a.tolist()) & set(b.tolist())) / k for a, b in zip(i4, i3)]
    )
    same4 = np.mean(truth[i4] == truth[:, None])
    same3 = np.mean(truth[i3] == truth[:, None])
    return {
        "n": len(original),
        "knn_overlap": float(overlap),
        "same_track_4d": float(same4),
        "same_track_pca3": float(same3),
    }


def _pair_distance_error(
    df: pl.DataFrame, pca: PCA, sample_pairs: int, seed: int = 42
) -> dict:
    rng = np.random.default_rng(seed)
    keys = df.select(["seed", "event_id"]).unique().sample(
        fraction=1.0, shuffle=True, seed=seed
    ).head(64)
    groups = {}
    for s, e in keys.iter_rows():
        groups[(s, e)] = df.filter(
            (pl.col("seed") == s) & (pl.col("event_id") == e)
        )
    available = [k for k, v in groups.items() if v.height >= 2]
    errors = []
    ratios = []
    while len(errors) < sample_pairs:
        key = available[int(rng.integers(len(available)))]
        x = _coordinates(groups[key])
        a, b = rng.choice(len(x), size=2, replace=False)
        d4 = float(np.linalg.norm(x[a] - x[b]))
        if d4 < 1.0e-8:
            continue
        y = pca.transform(x[[a, b]])[:, :3]
        d3 = float(np.linalg.norm(y[0] - y[1]))
        errors.append(abs(d3 - d4) / d4)
        ratios.append(d3 / d4)
    return {
        "pairs": sample_pairs,
        "median_relative_distance_error": float(np.median(errors)),
        "p95_relative_distance_error": float(np.quantile(errors, 0.95)),
        "median_distance_ratio_3d_over_4d": float(np.median(ratios)),
    }


def _plot_pca(out: Path, arm: str, evr: np.ndarray, evr_high_beta: np.ndarray):
    x = np.arange(1, 5)
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(x - width / 2, evr * 100, width, label="All cached hits")
    ax.bar(x + width / 2, evr_high_beta * 100, width, label=r"$\beta > 0.8$")
    ax.plot(x, np.cumsum(evr) * 100, "o-", color="black", label="All-hit cumulative")
    ax.set(
        xlabel="Principal component",
        ylabel="Explained variance (%)",
        title=f"{arm}: variance of learned 4-D clustering coordinates",
        xticks=x,
        ylim=(0, 105),
    )
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{arm}_pca_scree.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _umap_events(
    df: pl.DataFrame,
    pca: PCA,
    out: Path,
    arm: str,
    n_events: int,
) -> tuple[list[dict], str | None]:
    try:
        import umap
    except Exception as exc:  # analysis still yields PCA caches without UMAP
        return [], f"{type(exc).__name__}: {exc}"

    selected = df.filter(pl.col("beta") > 0.1)
    keys = _event_keys(selected, n_events)
    fig, axes = plt.subplots(len(keys), 2, figsize=(9.0, 3.6 * len(keys)))
    axes = np.atleast_2d(axes)
    reports = []
    for row, (seed, event_id) in enumerate(keys):
        sub = selected.filter(
            (pl.col("seed") == seed) & (pl.col("event_id") == event_id)
        )
        # UMAP is a topology diagnostic, not a truth-assisted evaluator. Limit
        # only the display cost with a deterministic row sample.
        if sub.height > 3000:
            sub = sub.sample(n=3000, seed=seed * 100000 + event_id)
        x4 = _coordinates(sub)
        x3 = pca.transform(x4)[:, :3].astype(np.float32)
        y = sub["mc_index"].to_numpy()
        z4 = umap.UMAP(
            n_components=2, n_neighbors=20, min_dist=0.05,
            metric="euclidean", random_state=42, n_jobs=1,
        ).fit_transform(x4)
        z3 = umap.UMAP(
            n_components=2, n_neighbors=20, min_dist=0.05,
            metric="euclidean", random_state=42, n_jobs=1,
        ).fit_transform(x3)
        metrics = _neighbour_metrics(x4, x3, y)
        metrics.update({"seed": seed, "event_id": event_id})
        reports.append(metrics)
        for col, (z, title) in enumerate(
            ((z4, "UMAP from 4-D"), (z3, "UMAP from PCA-3D"))
        ):
            ax = axes[row, col]
            ax.scatter(
                z[:, 0], z[:, 1], c=(y % 20), cmap="tab20",
                s=5, alpha=0.75, linewidths=0,
            )
            ax.set_title(
                f"{title} — seed {seed}, event {event_id}\n"
                f"{len(np.unique(y))} truth IDs, {len(y)} hits"
            )
            ax.set_xlabel("UMAP coordinate 1")
            ax.set_ylabel("UMAP coordinate 2")
    fig.suptitle(
        f"{arm}: event-local topology before and after dropping PCA component 4",
        y=1.002,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{arm}_umap_4d_vs_pca3.{ext}", dpi=180,
                    bbox_inches="tight")
    plt.close(fig)
    return reports, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--test-cache")
    parser.add_argument("--validation-mc", required=True)
    parser.add_argument("--test-mc")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--umap-events", type=int, default=4)
    parser.add_argument("--distance-pairs", type=int, default=20000)
    parser.add_argument(
        "--reuse-projected", action="store_true",
        help="Keep existing PCA-3D parquet caches while regenerating diagnostics.",
    )
    args = parser.parse_args()
    if bool(args.test_cache) != bool(args.test_mc):
        parser.error("--test-cache and --test-mc must be provided together")

    out = Path(args.output_root) / args.arm
    out.mkdir(parents=True, exist_ok=True)
    validation = pl.read_parquet(args.validation_cache)
    x = _coordinates(validation)
    pca = PCA(n_components=4, svd_solver="full").fit(x)
    high = validation.filter(pl.col("beta") > 0.8)
    pca_high = PCA(n_components=4, svd_solver="full").fit(_coordinates(high))

    projected_validation = (
        Path(args.output_root) / "validation" / args.arm / "emb"
        / "forward_hits.parquet"
    )
    projected_test = None
    if args.test_cache:
        projected_test = (
            Path(args.output_root) / "test" / args.arm / "emb"
            / "forward_hits.parquet"
        )
    if args.reuse_projected and projected_validation.exists():
        n_val = pl.scan_parquet(projected_validation).select(pl.len()).collect().item()
    else:
        n_val = _project_cache(args.validation_cache, projected_validation, pca)
    n_test = None
    if projected_test is not None:
        if args.reuse_projected and projected_test.exists():
            n_test = (
                pl.scan_parquet(projected_test).select(pl.len()).collect().item()
            )
        else:
            n_test = _project_cache(args.test_cache, projected_test, pca)
    mc_destinations = [
        (
            args.validation_mc,
            Path(args.output_root) / "validation" / args.arm
            / "mc_signal.parquet",
        )
    ]
    if args.test_mc:
        mc_destinations.append(
            (
                args.test_mc,
                Path(args.output_root) / "test" / args.arm
                / "mc_signal.parquet",
            )
        )
    for source, destination in mc_destinations:
        if not (args.reuse_projected and destination.exists()):
            destination.parent.mkdir(parents=True, exist_ok=True)
            pl.read_parquet(source).write_parquet(destination, compression="zstd")

    _plot_pca(out, args.arm, pca.explained_variance_ratio_,
              pca_high.explained_variance_ratio_)
    umap_reports, umap_error = _umap_events(
        validation, pca, out, args.arm, args.umap_events
    )
    report = {
        "arm": args.arm,
        "fit_split": "validation",
        "validation_rows": n_val,
        "pca_mean": pca.mean_.tolist(),
        "pca_components": pca.components_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance_first_3": float(
            pca.explained_variance_ratio_[:3].sum()
        ),
        "fourth_component_variance": float(pca.explained_variance_ratio_[3]),
        "participation_ratio": float(
            1.0 / np.square(pca.explained_variance_ratio_).sum()
        ),
        "high_beta_explained_variance_ratio":
            pca_high.explained_variance_ratio_.tolist(),
        "distance_preservation": _pair_distance_error(
            validation, pca, args.distance_pairs
        ),
        "event_neighbour_metrics": umap_reports,
        "umap_error": umap_error,
        "projected_validation_cache": str(projected_validation),
    }
    if n_test is not None:
        report["test_rows"] = n_test
        report["projected_test_cache"] = str(projected_test)
    with open(out / "embedding_dimension_report.json", "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
