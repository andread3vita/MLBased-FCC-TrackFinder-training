"""Clustering parameter sweep for v33 CGATr (no skip, 8D embed, secondaries as noise).

Usage:
    cd model_training
    PYTHONPATH=. python src/eval_sweep_v33.py \
        --data_dir /home/marko.cechovic/cgatr/data_parquet_train \
        --checkpoint checkpoints/cgatr_v33/cgatr_best.pt \
        --embed_dim 8 \
        --eval_seeds 181-190 --max_events 5000 \
        --output_dir eval_results/v33
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
from src.dataset.parquet_dataset import IDEAParquetDataset


# ---------------------------------------------------------------------------
# Model (must match train_cgatr_parquet-v26_strict_oc.py exactly)
# ---------------------------------------------------------------------------
class CGATrParquetModel(nn.Module):
    def __init__(self, hidden_mv_channels=16, hidden_s_channels=64,
                 num_blocks=10, normalize_mv_inputs=True,
                 embed_dim=8, beta_mlp=False):
        super().__init__()
        self._normalize = normalize_mv_inputs
        self.embed_dim = embed_dim
        self.bn_pos = nn.BatchNorm1d(3, momentum=0.1)
        self.bn_wire = nn.BatchNorm1d(3, momentum=0.1)
        self.bn_drift = nn.BatchNorm1d(1, momentum=0.1)

        gp_sparse = torch.load("cga_utils/cga_geometric_product.pt", weights_only=False)
        self.register_buffer("basis_gp", gp_sparse.to_dense().to(dtype=torch.float32))

        op_sparse = torch.load("cga_utils/cga_outer_product.pt", weights_only=False)
        self.register_buffer("basis_outer", op_sparse.to_dense().to(dtype=torch.float32))

        metadata = torch.load("cga_utils/cga_metadata.pt", weights_only=False)
        _DualCache.init_from_metadata(metadata, device=torch.device("cpu"))

        pin_basis = _compute_pin_equi_linear_basis(device=torch.device("cpu"), dtype=torch.float32)
        basis_q, basis_k = _build_dist_basis(device=torch.device("cpu"), dtype=torch.float32)
        basis_ip_weights = compute_inner_product_mask(self.basis_gp, device=torch.device("cpu"))

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

        self.clustering = nn.Linear(32, embed_dim, bias=False)
        if beta_mlp:
            self.beta = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
        else:
            self.beta = nn.Linear(32, 1)

    def forward(self, features, seq_lens):
        from xformers.ops.fmha import BlockDiagonalMask

        pos = features[:, :3]
        hit_type = features[:, 3:4]
        pos_normed = self.bn_pos(pos)

        is_vtx = (hit_type.squeeze(-1) == 0)
        is_dc = (hit_type.squeeze(-1) == 1)

        mv = torch.zeros(features.shape[0], 32, device=features.device, dtype=features.dtype)

        if is_vtx.any():
            mv[is_vtx] = embed_point(pos_normed[is_vtx])

        if is_dc.any():
            dc = features[is_dc]
            wire_normed = self.bn_wire(dc[:, 4:7])
            drift_normed = self.bn_drift(dc[:, 7:8]).squeeze(-1)
            wire_pos = wire_normed
            cos_s = torch.cos(dc[:, 9])
            sin_s = torch.sin(dc[:, 9])
            cos_a = torch.cos(dc[:, 8])
            sin_a = torch.sin(dc[:, 8])
            wire_dir = torch.stack([sin_s * cos_a, sin_s * sin_a, cos_s], dim=-1)
            wire_dir = wire_dir / (torch.norm(wire_dir, dim=-1, keepdim=True) + 1e-8)
            mv[is_dc] = embed_circle_ipns(wire_pos, wire_dir, drift_normed, self.basis_outer)

        mv = mv + embed_scalar(hit_type)

        if self._normalize:
            mv_norm = torch.norm(mv, dim=-1, keepdim=True).clamp(min=1e-6)
            mv = mv / mv_norm

        mv = mv.unsqueeze(1)

        mask = BlockDiagonalMask.from_seqlens(seq_lens)
        out_mv, _ = self.cgatr(mv, scalars=None, attention_mask=mask)
        out = out_mv[:, 0, :]
        return torch.cat([self.clustering(out), self.beta(out)], dim=1)


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
    """Run model inference on dataset, return cached per-event results.

    Caches signal-only hits for clustering sweep AND tracks noise/secondary
    beta statistics to verify noise suppression.
    """
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
    """Sweep DBSCAN over eps x min_samples grid."""
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
    """Sweep HDBSCAN over min_cluster_size values."""
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
    """Average metrics across events."""
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
    parser = argparse.ArgumentParser(description="Clustering sweep for v33 CGATr")
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

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"embed_dim={args.embed_dim} | beta_mlp={args.beta_mlp} | cosine_norm={args.cosine_norm}")
    model = CGATrParquetModel(
        hidden_mv_channels=args.hidden_mv_channels,
        hidden_s_channels=args.hidden_s_channels,
        num_blocks=args.num_blocks,
        embed_dim=args.embed_dim,
        beta_mlp=args.beta_mlp,
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model = model.to(device)
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # --- Load data ---
    start, end = parse_seed_range(args.eval_seeds)
    print(f"Loading eval data: seeds {start}-{end - 1}, max_hits={args.max_hits}")
    dataset = IDEAParquetDataset(args.data_dir, seed_range=(start, end),
                                  max_hits_per_event=args.max_hits)
    print(f"Dataset: {len(dataset)} events")

    # --- Inference ---
    print("\n=== Running inference ===")
    cached, noise_betas_list = run_inference(model, dataset, device, args.max_events,
                                              embed_dim=args.embed_dim, cosine_norm=args.cosine_norm)

    # --- Embedding stats ---
    all_coords = np.concatenate([e["coords"] for e in cached])
    print(f"\nEmbedding stats ({len(all_coords)} signal hits across {len(cached)} events):")
    print(f"  Range: [{all_coords.min():.3f}, {all_coords.max():.3f}]")
    print(f"  Std per dim: {all_coords.std(axis=0)}")
    print(f"  Mean norm: {np.linalg.norm(all_coords, axis=1).mean():.3f}")

    # --- Signal beta stats ---
    all_betas = np.concatenate([e["beta"] for e in cached])
    print("\nSignal beta distribution:")
    for threshold in [0.1, 0.2, 0.3, 0.5, 0.7]:
        pct = (all_betas > threshold).mean() * 100
        print(f"  beta > {threshold}: {pct:.1f}%")

    # --- Noise beta stats (secondary + mc_index==0 hits) ---
    if noise_betas_list:
        all_noise_betas = np.concatenate(noise_betas_list)
        print(f"\nNoise/secondary beta distribution ({len(all_noise_betas)} hits):")
        for threshold in [0.1, 0.2, 0.3, 0.5, 0.7]:
            pct = (all_noise_betas < threshold).mean() * 100
            print(f"  beta < {threshold}: {pct:.1f}%")
        print(f"  mean noise beta: {all_noise_betas.mean():.4f}")
        print(f"  median noise beta: {np.median(all_noise_betas):.4f}")

    all_results = []

    # --- Beta-greedy sweep ---
    print("\n=== Beta-greedy sweep ===")
    tbeta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    td_values = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    all_results.extend(sweep_greedy(cached, tbeta_values, td_values))

    # --- DBSCAN sweep ---
    print("\n=== DBSCAN sweep ===")
    eps_values = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    min_samples_values = [2, 3, 5]
    all_results.extend(sweep_dbscan(cached, eps_values, min_samples_values))

    # --- HDBSCAN sweep ---
    print("\n=== HDBSCAN sweep ===")
    min_cluster_sizes = [3, 5, 10, 15]
    all_results.extend(sweep_hdbscan(cached, min_cluster_sizes))

    # --- Sort by match rate and print summary table ---
    all_results.sort(key=lambda r: r["match_rate"], reverse=True)

    print(f"\n{'='*80}")
    print(f"SWEEP RESULTS — sorted by match rate ({len(all_results)} configs)")
    print(f"{'='*80}")
    print(f"{'Rank':>4}  {'Algorithm':<14} {'Params':<32} {'Purity':>7} {'Effic':>7} {'Match':>7}")
    print(f"{'-'*80}")
    for i, r in enumerate(all_results[:30]):
        print(f"{i+1:>4}  {r['algorithm']:<14} {r['params']:<32} "
              f"{r['purity']:>7.3f} {r['efficiency']:>7.3f} {r['match_rate']:>7.3f}")

    if len(all_results) > 30:
        print(f"  ... ({len(all_results) - 30} more configs omitted)")

    # --- Best per algorithm ---
    print(f"\n{'='*80}")
    print("BEST PER ALGORITHM")
    print(f"{'='*80}")
    for algo in ["beta-greedy", "DBSCAN", "HDBSCAN"]:
        algo_results = [r for r in all_results if r["algorithm"] == algo]
        if algo_results:
            best = algo_results[0]
            print(f"  {algo:<14} {best['params']:<32} "
                  f"pur={best['purity']:.3f} eff={best['efficiency']:.3f} match={best['match_rate']:.3f}")

    # --- Save JSON ---
    output_path = os.path.join(args.output_dir, "sweep_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "eval_seeds": args.eval_seeds,
            "max_hits": args.max_hits,
            "n_events": len(cached),
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
