"""Forward-pass exporter for a GGTF-recipe checkpoint (ExampleWrapper / Gatr_withModifications).

Writes the same per-hit schema our own `src/eval/forward_pass.py` writes for the CGA arm --
event_id, seed, hit_order, coord_0..coord_2, beta, mc_index, n_hits_total -- so both arms are
scored by the identical `src/eval/ggtf_full_eval.py` (with --embed-dim 3 for this arm, since
Gatr_withModifications.output_dim=4 gives a 3-dim clustering coordinate plus beta).

`mc_index` here is the RAW mc_index column from the parquet (`particle_number_nomap_original`
in the adapter), not GGTF's own per-event 1..K relabeling (`particle_number`): the former is
truth that matches `mc_signal.parquet` and every other arm's convention, the latter is an
internal training convenience that would silently break every join downstream.

Must run inside the ggtf-gatr:v9-cgatr container from /work/model_training/GATR --
`load_basis()` reads two files by relative path and hardcodes cuda, so this needs a GPU and
that exact working directory.

    python -m src.models.export_ggtf_forward \
        --data_dir /data --seeds 1-10 --checkpoint /ckpt/ab_11.ckpt --out /out/forward_hits.parquet
"""
from __future__ import annotations

import argparse
import types

import dgl
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.dataset.parquet_ggtf_adapter import ParquetGGTFDataset, parse_seed_range
from src.models.Gatr_withModifications import ExampleWrapper

# Attributes rebuilt by load_basis() in __init__ from files on disk, not learned, so they are
# never in the checkpoint's state_dict; anything else missing or unexpected is a real mismatch.
_DERIVED_ATTRS = ("basis_gp", "basis_outer", "pin_basis", "basis_q", "basis_k", "basis_gp_mask")


def build_input(g) -> torch.Tensor:
    """The 7 columns the model reads: position, hit type, drift displacement."""
    return torch.cat(
        (g.ndata["pos_hits_xyz"], g.ndata["hit_type"].view(-1, 1), g.ndata["vector"]),
        dim=1,
    )


def load_model(checkpoint_path: str, device: torch.device) -> ExampleWrapper:
    # Only accessed by configure_optimizers/*_step, which this script never calls, so the
    # values are placeholders matching the smoke test's pattern rather than the real recipe.
    args = types.SimpleNamespace(start_lr=1e-3, predict=False, capacity_matched=True,
                                  loss_computation="dense")
    model = ExampleWrapper(args).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if not k.startswith(_DERIVED_ATTRS)]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match ExampleWrapper: missing={bad_missing}, "
            f"unexpected={list(unexpected)}"
        )
    model.eval()
    return model


def seed_of(dataset: ParquetGGTFDataset, i: int) -> tuple[int, int]:
    seed_dir, event_id = dataset._index[i]
    seed = int(seed_dir.rstrip("/").rsplit("_", 1)[-1])
    return seed, event_id


_SCHEMA = pa.schema([
    ("event_id", pa.int64()), ("seed", pa.int64()), ("hit_order", pa.int64()),
    ("coord_0", pa.float32()), ("coord_1", pa.float32()), ("coord_2", pa.float32()),
    ("beta", pa.float32()), ("mc_index", pa.int64()), ("n_hits_total", pa.int64()),
])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--seeds", required=True, help="'1-100' inclusive")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_events", type=int, default=8,
                     help="events per forward batch. Measured 8 as fast (7.7 ev/s) and safe "
                          "sharing a GPU with active training (~13GB used); 32 pushed a "
                          "contended V100 to 31GB/32GB and the run slowed by >30x, likely "
                          "allocator thrashing rather than genuine compute -- raise only on "
                          "an otherwise-idle GPU.")
    a = ap.parse_args()

    device = torch.device("cuda")
    model = load_model(a.checkpoint, device)
    ds = ParquetGGTFDataset(a.data_dir, parse_seed_range(a.seeds))
    n = len(ds)
    print(f"loaded {n} events from {a.data_dir} seeds {a.seeds}", flush=True)

    writer = pq.ParquetWriter(a.out, _SCHEMA)
    with torch.no_grad():
        for start in range(0, n, a.batch_events):
            idx = list(range(start, min(start + a.batch_events, n)))
            samples = [ds[i] for i in idx]
            metas = [seed_of(ds, i) for i in idx]
            graphs = [g for g, _ in samples]
            bg = dgl.batch(graphs).to(device)
            inp = build_input(bg)
            out = model(bg, inp)  # [N, 4] = coord0, coord1, coord2, beta

            bg.ndata["_coord0"] = out[:, 0]
            bg.ndata["_coord1"] = out[:, 1]
            bg.ndata["_coord2"] = out[:, 2]
            bg.ndata["_beta"] = out[:, 3]
            per_graph = dgl.unbatch(bg)

            rows = {c: [] for c in _SCHEMA.names}
            for (seed, event_id), gj in zip(metas, per_graph):
                mc = gj.ndata["particle_number_nomap_original"].cpu().numpy().astype(np.int64)
                n_hits_total_map = np.bincount(mc) if mc.size else np.zeros(0, dtype=np.int64)
                n_hits_total = n_hits_total_map[mc]
                n_hits = mc.shape[0]
                rows["event_id"].extend([event_id] * n_hits)
                rows["seed"].extend([seed] * n_hits)
                rows["hit_order"].extend(range(n_hits))
                rows["coord_0"].extend(gj.ndata["_coord0"].cpu().numpy().astype(np.float32).tolist())
                rows["coord_1"].extend(gj.ndata["_coord1"].cpu().numpy().astype(np.float32).tolist())
                rows["coord_2"].extend(gj.ndata["_coord2"].cpu().numpy().astype(np.float32).tolist())
                rows["beta"].extend(gj.ndata["_beta"].cpu().numpy().astype(np.float32).tolist())
                rows["mc_index"].extend(mc.tolist())
                rows["n_hits_total"].extend(n_hits_total.tolist())

            writer.write_table(pa.table(rows, schema=_SCHEMA))
            done = start + len(idx)
            if (start // a.batch_events) % 20 == 0:
                print(f"  {done}/{n} events", flush=True)

    writer.close()
    print(f"EXPORT_GGTF_FORWARD_DONE events={n} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
