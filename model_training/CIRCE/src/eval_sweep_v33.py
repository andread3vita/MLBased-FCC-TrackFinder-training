"""Clustering parameter sweep helpers for C-GATr FCC.

The forked CGATrParquetModel class has been removed from this file.
CGATrParquetModel is re-exported from src.model (the canonical one).
All clustering/metric/sweep helpers are kept verbatim.
"""

import os
import sys
import json
import argparse
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn

from src.dataset.parquet_dataset import IDEAParquetDataset

# Re-export canonical model — no fork in this package.
from src.model import CGATrParquetModel  # noqa: F401


# ---------------------------------------------------------------------------
# Clustering algorithms
# ---------------------------------------------------------------------------
def get_clustering_greedy(betas, X, tbeta=0.5, td=0.5):
    """Beta-greedy clustering (matches training eval)."""
    n_points = betas.shape[0]
    select = betas > tbeta
    indices = np.nonzero(select)[0]
    indices = indices[np.argsort(-betas[select])]
    unassigned = np.arange(n_points)
    clustering = -1 * np.ones(n_points, dtype=np.int32)
    for idx in indices:
        d = np.linalg.norm(X[unassigned] - X[idx], axis=-1)
        assigned = unassigned[d < td]
        clustering[assigned] = idx
        unassigned = unassigned[~(d < td)]
    return clustering


def get_clustering_dbscan(coords, eps=0.5, min_samples=3):
    from sklearn.cluster import DBSCAN
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)


def get_clustering_hdbscan(coords, min_cluster_size=5):
    try:
        from hdbscan import HDBSCAN
        return HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(coords)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Metrics (same logic as in training script)
# ---------------------------------------------------------------------------
def compute_metrics(pred_labels, mc_true):
    """Compute purity, efficiency, match rate from predicted labels and true mc_index."""
    unique_pred = np.unique(pred_labels[pred_labels >= 0])
    unique_true = np.unique(mc_true[mc_true >= 0])

    if len(unique_pred) == 0:
        return {"purity": 0.0, "efficiency": 0.0, "match_rate": 0.0,
                "n_pred_clusters": 0, "n_true_tracks": len(unique_true)}

    purities = []
    for pid in unique_pred:
        cluster_mc = mc_true[pred_labels == pid]
        if len(cluster_mc) > 0:
            purities.append(np.bincount(cluster_mc).max() / len(cluster_mc))

    matched = 0
    n_matchable = 0
    effs = []
    for tid in unique_true:
        tmask = mc_true == tid
        n_true = tmask.sum()
        if n_true < 2:
            continue
        n_matchable += 1
        pred_for_track = pred_labels[tmask]
        pred_for_track = pred_for_track[pred_for_track >= 0]
        if len(pred_for_track) == 0:
            effs.append(0.0)
            continue
        best_label, best_match = Counter(pred_for_track).most_common(1)[0]
        eff = best_match / n_true
        pur = best_match / (pred_labels == best_label).sum()
        effs.append(eff)
        if pur > 0.75:
            matched += 1

    return {
        "purity": float(np.mean(purities)) if purities else 0.0,
        "efficiency": float(np.mean(effs)) if effs else 0.0,
        "match_rate": matched / max(n_matchable, 1) if effs else 0.0,
        "n_pred_clusters": len(unique_pred),
        "n_true_tracks": len(unique_true),
    }


# ---------------------------------------------------------------------------
# Inference + caching
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(model, dataset, device, max_events, embed_dim=8, cosine_norm=False):
    """Run model inference on dataset, return cached per-event results."""
    model.eval()
    cached = []
    noise_betas_all = []
    n_events = min(max_events, len(dataset))

    for idx in range(n_events):
        event = dataset[idx]
        if event is None:
            continue

        features = event["features"].to(device)
        mc_index = event["mc_index"].numpy()
        is_secondary = event["is_secondary"].numpy().astype(bool)
        seq_lens = [event["n_hits"]]

        output = model(features, seq_lens)
        coords = output[:, :embed_dim].cpu().numpy()
        if cosine_norm:
            norms = np.linalg.norm(coords, axis=1, keepdims=True)
            coords = coords / np.clip(norms, 1e-6, None)
        beta = torch.sigmoid(output[:, embed_dim]).cpu().numpy()

        noise_mask = is_secondary | (mc_index == 0)
        if noise_mask.any():
            noise_betas_all.append(beta[noise_mask])

        sig_mask = (~is_secondary) & (mc_index != 0)
        if sig_mask.sum() < 4:
            continue

        cached.append({
            "coords": coords[sig_mask],
            "beta": beta[sig_mask],
            "mc_index": mc_index[sig_mask],
        })

        if (idx + 1) % 50 == 0:
            print(f"  Inference: {idx + 1}/{n_events} events", flush=True)

    print(f"  Inference: {n_events}/{n_events} events", flush=True)
    print(f"  Cached {len(cached)} events for sweep", flush=True)
    return cached, noise_betas_all


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def sweep_greedy(cached_events, tbeta_values, td_values):
    """Sweep beta-greedy clustering over tbeta x td grid."""
    results = []
    total = len(tbeta_values) * len(td_values)
    done = 0

    for tbeta in tbeta_values:
        for td in td_values:
            all_m = []
            for evt in cached_events:
                labels = get_clustering_greedy(evt["beta"], evt["coords"], tbeta=tbeta, td=td)
                all_m.append(compute_metrics(labels, evt["mc_index"]))

            avg = _aggregate(all_m)
            avg["algorithm"] = "beta-greedy"
            avg["params"] = f"tbeta={tbeta:.2f}, td={td:.2f}"
            avg["tbeta"] = tbeta
            avg["td"] = td
            results.append(avg)
            done += 1
            print(f"  greedy [{done}/{total}] tbeta={tbeta:.2f} td={td:.2f} | "
                  f"pur={avg['purity']:.3f} eff={avg['efficiency']:.3f} match={avg['match_rate']:.3f}",
                  flush=True)

    return results


def sweep_dbscan(cached_events, eps_values, min_samples_values):
    results = []
    total = len(eps_values) * len(min_samples_values)
    done = 0

    for eps in eps_values:
        for ms in min_samples_values:
            all_m = []
            for evt in cached_events:
                labels = get_clustering_dbscan(evt["coords"], eps=eps, min_samples=ms)
                all_m.append(compute_metrics(labels, evt["mc_index"]))

            avg = _aggregate(all_m)
            avg["algorithm"] = "DBSCAN"
            avg["params"] = f"eps={eps:.2f}, min_samples={ms}"
            avg["eps"] = eps
            avg["min_samples"] = ms
            results.append(avg)
            done += 1
            print(f"  DBSCAN [{done}/{total}] eps={eps:.2f} ms={ms} | "
                  f"pur={avg['purity']:.3f} eff={avg['efficiency']:.3f} match={avg['match_rate']:.3f}",
                  flush=True)

    return results


def sweep_hdbscan(cached_events, min_cluster_sizes):
    results = []

    for mcs in min_cluster_sizes:
        all_m = []
        skip = False
        for evt in cached_events:
            labels = get_clustering_hdbscan(evt["coords"], min_cluster_size=mcs)
            if labels is None:
                print("  HDBSCAN not installed, skipping", flush=True)
                skip = True
                break
            all_m.append(compute_metrics(labels, evt["mc_index"]))

        if skip:
            break

        avg = _aggregate(all_m)
        avg["algorithm"] = "HDBSCAN"
        avg["params"] = f"min_cluster_size={mcs}"
        avg["min_cluster_size"] = mcs
        results.append(avg)
        print(f"  HDBSCAN mcs={mcs} | "
              f"pur={avg['purity']:.3f} eff={avg['efficiency']:.3f} match={avg['match_rate']:.3f}",
              flush=True)

    return results


def _aggregate(metrics_list):
    return {
        "purity": float(np.mean([m["purity"] for m in metrics_list])),
        "efficiency": float(np.mean([m["efficiency"] for m in metrics_list])),
        "match_rate": float(np.mean([m["match_rate"] for m in metrics_list])),
        "n_events": len(metrics_list),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_seed_range(s):
    parts = s.split("-")
    return int(parts[0]), int(parts[1]) + 1


def main():
    parser = argparse.ArgumentParser(description="Clustering sweep for C-GATr FCC")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval_seeds", type=str, default="111-112")
    parser.add_argument("--max_hits", type=int, default=3000)
    parser.add_argument("--max_events", type=int, default=100)
    parser.add_argument("--num_blocks", type=int, default=10)
    parser.add_argument("--hidden_mv_channels", type=int, default=16)
    parser.add_argument("--hidden_s_channels", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=8)
    parser.add_argument("--beta_mlp", action="store_true", default=False)
    parser.add_argument("--cosine_norm", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="eval_results/v33")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = CGATrParquetModel(args)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model = model.to(device)
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    start, end = parse_seed_range(args.eval_seeds)
    dataset = IDEAParquetDataset(args.data_dir, seed_range=(start, end),
                                  max_hits_per_event=args.max_hits)
    print(f"Dataset: {len(dataset)} events")

    cached, noise_betas_list = run_inference(model, dataset, device, args.max_events,
                                              embed_dim=args.embed_dim, cosine_norm=args.cosine_norm)

    all_results = []
    tbeta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    td_values = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    all_results.extend(sweep_greedy(cached, tbeta_values, td_values))

    output_path = os.path.join(args.output_dir, "sweep_results.json")
    with open(output_path, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "eval_seeds": args.eval_seeds,
                   "n_events": len(cached), "results": all_results}, f, indent=2)
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
