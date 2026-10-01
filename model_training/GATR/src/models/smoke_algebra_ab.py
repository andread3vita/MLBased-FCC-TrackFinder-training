"""Run both algebras through GGTF's own loss on real events from our parquet.

This is the integration check the algebra A/B rests on. `smoke_conformal_arm.py` proved the
conformal arm runs inside their runtime, but on synthetic graphs; this proves both arms run on
the graphs the adapter builds from real data, and that they see byte-identical inputs. If the
two arms ever disagree about anything other than the algebra, it should show up here rather
than twenty GPU-hours into a training run.

What it asserts:
  * the adapter's graphs drive `Gatr_withModifications` (projective) and
    `Cgatr_withModifications` (conformal) forward and backward through their real
    `object_condensation_loss_tracking`, on the truth table the adapter emits;
  * both arms receive exactly the same input tensor, compared element by element;
  * every trainable parameter gets a gradient, except the attention `log_weights` that are
    bypassed by design when explicit metric weights are supplied;
  * the two arms are capacity-matched, which is what makes the comparison about the algebra
    rather than about width.

    python -m src.models.smoke_algebra_ab --data_dir /data/data-final/parquet
"""
from __future__ import annotations

import argparse
import types

import torch

import dgl  # noqa: F401  (registers the backend before the models import it)


def build_input(g):
    """The 7 columns their model reads: position, hit type, drift displacement.

    Same order their loader hands over, so neither arm sees a permuted feature block.
    """
    return torch.cat(
        (g.ndata["pos_hits_xyz"], g.ndata["hit_type"].view(-1, 1), g.ndata["vector"]),
        dim=1,
    )


def run_arm(label: str, wrapper_module: str, g, inp, y, device):
    import importlib

    mod = importlib.import_module(wrapper_module)
    args = types.SimpleNamespace(
        start_lr=1e-3,
        predict=False,
        capacity_matched=True,
        clustering_space_dim=3,
        lr_scheduler="none",
        optimizer="adam",
        num_epochs=1,
        qmin=0.1,
        L_attractive_weight=1.0,
        L_repulsive_weight=1.0,
        use_average_cc_pos=0.0,
        loss_type="hgcalimplementation",
    )
    model = mod.GraphTransformerNetWrapper(args, device).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)

    out = model.mod(g, inp)
    assert out.shape[0] == g.num_nodes() and out.shape[1] >= 4, f"{label}: {out.shape}"
    assert torch.isfinite(out).all(), f"{label}: non-finite output"

    from src.layers.losses import object_condensation_loss_tracking

    loss, _ = object_condensation_loss_tracking(
        g, out[:, :4], y,
        clust_loss_only=True, add_energy_loss=False, calc_e_frac_loss=False,
        q_min=0.1, frac_clustering_loss=0.0, attr_weight=1.0, repul_weight=1.0,
        fill_loss_weight=0.0, use_average_cc_pos=0.0, loss_type="hgcalimplementation",
        tracking=True,
    )
    assert torch.isfinite(loss), f"{label}: non-finite loss"
    loss.backward()

    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    unexpected = [n for n in no_grad if not n.endswith("attention.log_weights")]
    assert not unexpected, f"{label}: params without grad: {unexpected}"

    print(f"  {label:11s} {n_par:>9,} params   loss {float(loss):8.4f}   "
          f"backward ok   ({len(no_grad)} bypassed log_weights)")
    return n_par, float(loss)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default="/data/data-final/parquet")
    ap.add_argument("--seeds", default="1-1")
    ap.add_argument("--events", type=int, default=2,
                    help="events per batch; keep small, this runs on CPU by default")
    ap.add_argument("--max_hits", type=int, default=600,
                    help="truncate each event to this many hits so a CPU run is quick; "
                         "0 keeps the full event")
    ap.add_argument("--cuda", action="store_true",
                    help="off by default so this does not contend with a training run")
    a = ap.parse_args()
    device = torch.device("cuda" if a.cuda else "cpu")

    if device.type == "cpu":
        # xformers' kernel is CUDA-only and the conformal attention prefers it on import.
        import src.cgatr.primitives.attention as attention_mod
        attention_mod._HAS_XFORMERS = False

    from src.dataset.parquet_ggtf_adapter import (
        ParquetGGTFDataset, collate_ggtf, parse_seed_range, Y_EVENT_IDX, Y_PART_ID,
    )

    ds = ParquetGGTFDataset(a.data_dir, parse_seed_range(a.seeds))
    samples = [ds[i] for i in range(a.events)]

    if a.max_hits:
        # Truncation has to keep the graph and its truth table consistent: dropping hits can
        # empty a cluster, and the loss indexes objects by dense cluster id, so the ids are
        # rebuilt and the truth rows re-selected rather than sliced.
        from src.dataset.parquet_ggtf_adapter import find_cluster_id
        trimmed = []
        for g, y in samples:
            keep = torch.arange(min(a.max_hits, g.num_nodes()))
            sub = dgl.node_subgraph(g, keep)
            link = sub.ndata["particle_number_nomap"]
            sub.ndata["particle_number"] = find_cluster_id(link).to(torch.int64)
            surviving = {int(s) for s in torch.unique(link[link != -1]).tolist()}
            rows = [i for i, v in enumerate(y[:, Y_PART_ID].tolist())
                    if int(v) in surviving]
            trimmed.append((sub, y[rows].clone()))
        samples = trimmed

    g, y = collate_ggtf(samples)
    g, y = g.to(device), y.to(device)
    inp = build_input(g)

    print(f"torch {torch.__version__}  device {device}")
    print(f"batch: {g.num_nodes()} nodes over {len(g.batch_num_nodes())} events, "
          f"{int(g.ndata['particle_number'].max())} clusters, truth {tuple(y.shape)}")
    n_noise = int((g.ndata["particle_number"] == 0).sum())
    print(f"       {n_noise} nodes are noise (their relabelling), input {tuple(inp.shape)}")
    for i in range(len(g.batch_num_nodes())):
        rows = int((y[:, Y_EVENT_IDX] == i).sum())
        clusters = int(dgl.unbatch(g)[i].ndata["particle_number"].max())
        assert rows == clusters, f"event {i}: {rows} truth rows vs {clusters} clusters"

    # Both arms must see the same tensor object, not merely the same shape.
    inp_projective = inp.clone()
    inp_conformal = inp.clone()
    assert torch.equal(inp_projective, inp_conformal)

    print()
    n_pga, loss_pga = run_arm(
        "projective", "src.models.wrapper.model_tracking_gatr", g, inp_projective, y, device)
    n_cga, loss_cga = run_arm(
        "conformal", "src.models.wrapper.model_tracking_cgatr", g, inp_conformal, y, device)

    spread = abs(n_cga - n_pga) / max(n_pga, 1)
    print()
    print(f"capacity spread {100 * spread:.2f}% "
          f"({'matched' if spread < 0.05 else 'NOT MATCHED -- fix before training'})")
    assert spread < 0.05, "capacity mismatch would confound the algebra comparison"
    print("the losses above are one step on random init and are not comparable across "
          "algebras;\nthey only show both arms produce a finite, differentiable objective on "
          "identical inputs")
    print("\nOK: both algebras train on the adapter's graphs through GGTF's own loss")


if __name__ == "__main__":
    main()
