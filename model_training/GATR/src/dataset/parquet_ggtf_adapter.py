"""Our IDEA parquet events in GGTF's DGL graph contract.

Why this exists: the algebra A/B has to run on one dataset and one target definition, or it
measures the data pipeline rather than the algebra. Their loader reads an HDF5 layout produced
by their own condor pipeline, which we do not have; we have parquet. This module produces the
graph their model and loss consume, from our parquet, so `Gatr_withModifications` (projective)
and `Cgatr_withModifications` (conformal) can be trained on byte-identical inputs.

The contract is taken from `functions_graph_tracking.create_graph_tracking_global` in the
`get_vtx=True, vector_like_data=True` branch, which is the one their configuration selects:

  * one node per hit -- a vertex hit at its own position, a drift-chamber hit at its **left
    tangency point**, not at the wire and not two nodes per hit. The `vector=False` branch of
    their loader does emit two nodes per drift hit; that is a different encoding and not the
    one in use.
  * `vector` carries `right - left` for drift hits and exactly zero for vertex hits, so the
    drift geometry reaches the network as a displacement rather than a position.
  * vertex hits come first, then drift hits, matching their `cat((x[mask_vtx], x[mask_dc]))`.
  * `hit_type` is 0 for drift and 1 for vertex. Our parquet already uses that convention, so
    the column passes through unmapped -- verified, not assumed.
  * `particle_number` is a per-event contiguous cluster id where **0 means noise** and real
    particles are 1..K, per their `find_cluster_id`.

Truth (`y`) is per-particle, one row per surviving cluster in cluster-id order. Their loss
reads three columns and no others (`calc_LV_Lbeta` -> `calculate_delta_MC`): column 0 as theta,
column 1 as phi, and the **last** column as an event index within the batch. It uses them for a
nearest-neighbour separation in (eta, phi) that weights the repulsive term. The remaining
columns are carried for readability and for the physics cuts their notebook applies.

Filtering follows `create_garbage_label(minNumHits=3)`, which **relabels rather than deletes**:
a hit belonging to a particle with fewer than three hits stays in the graph and becomes noise.
That distinction is the whole point -- our own `--drop_loopers` deletes hits, which is a
different operation on a different target set, and conflating the two is what made an earlier
round of this work compare against the wrong baseline.

Their secondary criterion is a no-op on our data: `produced_by_secondary` is all zero in
`data-final/parquet`, verified across the files. It is implemented anyway so the code does not
quietly depend on that.

Self-test:
    python -m src.dataset.parquet_ggtf_adapter --data_dir <parquet> --seeds 1-2
"""
from __future__ import annotations

import argparse
import glob
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import random

import torch
from torch.utils.data import Dataset, Sampler

import dgl
import pyarrow.parquet as pq


# Truth column layout, copied from `particle_features` in their `config_tracking.yaml` rather
# than invented, so their own evaluation code can read this table without a translation step.
# Their loss only touches THETA, PHI and the last column, but anything of theirs that reads
# gen_status expects it at index 7, and a plausible-looking reordering would be silent.
Y_THETA, Y_PHI, Y_MASS, Y_PDG, Y_PART_ID = 0, 1, 2, 3, 4
Y_P, Y_PT, Y_GEN_STATUS, Y_PARENT = 5, 6, 7, 8
Y_NCOLS = 9

# Appended by the collate, exactly as their `add_batch_number` does, so it lands last where
# `calculate_delta_MC` reads it as `y[:, -1]`. Their per-event table has no such column; it
# only exists once events are batched.
Y_EVENT_IDX = 9

# Their config carries no vertex position, so vertex-radius cuts (their notebook uses R < 50 mm)
# have to read the parquet separately rather than expecting a column here. Deliberate: matching
# their layout matters more than saving that read.


def parse_seed_range(spec: str) -> Tuple[int, int]:
    """'1-60' or '7' -> inclusive (lo, hi)."""
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return int(lo), int(hi)
    v = int(spec)
    return v, v


class _Table:
    """One parquet file as column arrays plus a row range per event.

    pyarrow rather than polars because the container ships pyarrow 12 on Python 3.8 and no
    polars, and adding one only to re-do what pyarrow already does would mean rebuilding the
    image the A/B runs in. Rows are sorted once by `event_id` so an event is a contiguous
    slice, which is cheaper than materialising a frame per event.
    """

    def __init__(self, path: str, columns: List[str]):
        table = pq.read_table(path, columns=columns)
        # ChunkedArray.to_numpy takes no arguments in pyarrow 12; the zero_copy_only kwarg
        # belongs to Array. Copying is what we want here anyway, since the result is sorted.
        arrays = {c: table.column(c).to_numpy() for c in columns}
        order = np.argsort(arrays["event_id"], kind="stable")
        self.cols: Dict[str, np.ndarray] = {c: a[order] for c, a in arrays.items()}
        ids = self.cols["event_id"]
        uniq, starts = np.unique(ids, return_index=True)
        stops = np.append(starts[1:], len(ids))
        self.rows: Dict[int, Tuple[int, int]] = {
            int(e): (int(s), int(t)) for e, s, t in zip(uniq, starts, stops)
        }

    def event(self, event_id: int) -> Dict[str, np.ndarray]:
        lo, hi = self.rows[event_id]
        return {c: a[lo:hi] for c, a in self.cols.items()}


@lru_cache(maxsize=4)
def _read_seed(seed_dir: str) -> Tuple[_Table, _Table, _Table]:
    """Read one seed's three tables. Cached, since events are visited in seed order.

    maxsize is deliberately small: each entry holds a whole seed's hits.
    """
    def load(pattern: str, columns: List[str]) -> _Table:
        matches = glob.glob(os.path.join(seed_dir, pattern))
        if not matches:
            raise FileNotFoundError(f"no {pattern} under {seed_dir}")
        return _Table(matches[0], columns)

    dc = load("dc_hits_*.parquet", [
        "left_x", "left_y", "left_z", "right_x", "right_y", "right_z",
        "mc_index", "produced_by_secondary", "hit_type", "event_id",
    ])
    vtx = load("vtx_hits_*.parquet", [
        "hit_x", "hit_y", "hit_z",
        "mc_index", "produced_by_secondary", "hit_type", "event_id",
    ])
    mc = load("mc_particles_*.parquet", [
        "mc_index", "theta", "phi", "mass", "pdg", "p", "pt", "gen_status",
        "parent_index", "event_id",
    ])
    return dc, vtx, mc


def find_cluster_id(hit_particle_link: torch.Tensor) -> torch.Tensor:
    """Their `find_cluster_id`: noise (-1) becomes 0, particles become 1..K in sorted order.

    Reimplemented rather than imported so this module does not drag in their loader's
    numpy/HDF5 dependencies, and kept deliberately equivalent -- the 1-based offset with 0
    reserved for noise is what their loss assumes when it treats cluster 0 as background.
    """
    is_noise = hit_particle_link == -1
    cluster_id = torch.zeros_like(hit_particle_link)
    if is_noise.all():
        return cluster_id
    signal_links = hit_particle_link[~is_noise]
    uniques = torch.unique(signal_links)
    cluster_id[~is_noise] = torch.searchsorted(uniques, signal_links) + 1
    return cluster_id


def apply_garbage_label(
    hit_particle_link: torch.Tensor,
    is_secondary: torch.Tensor,
    min_num_hits: int = 3,
) -> torch.Tensor:
    """Their `create_garbage_label`, expressed as relabelling to noise.

    Their function returns keep-masks that the caller uses to move hits out of the target set
    while leaving them in the graph. Relabelling the link to -1 has exactly that effect here,
    and makes the "hits stay, targets go" semantics explicit at the call site.

    A hit becomes noise if it is itself flagged secondary, or if its particle has fewer than
    `min_num_hits` hits, or if every one of its particle's hits is secondary (their extra loop,
    which is subsumed by the per-hit rule but kept so the predicate reads the same).
    """
    link = hit_particle_link.clone()
    noise = is_secondary.bool().clone()

    signal = link[~noise] if noise.any() else link
    if signal.numel() > 0:
        uniques, counts = torch.unique(link, return_counts=True)
        too_few = uniques[counts < min_num_hits]
        if too_few.numel() > 0:
            noise |= torch.isin(link, too_few)

    link[noise] = -1
    return link


class ParquetGGTFDataset(Dataset):
    """One item is one event: an edgeless DGL graph plus its per-particle truth table.

    Edgeless is faithful, not a shortcut: their loader builds a bare `dgl.DGLGraph()` and
    attaches only node data, and neither their model nor their loss touches `edata` or does
    message passing -- the transformer attends over a block-diagonal mask derived from
    `batch_num_nodes()`. Verified by inspection of both models and `object_cond.py`.
    """

    def __init__(
        self,
        data_dir: str,
        seed_range: Tuple[int, int],
        min_num_hits: int = 3,
        garbage_label: bool = True,
        max_events_per_seed: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.min_num_hits = min_num_hits
        self.garbage_label = garbage_label

        lo, hi = seed_range
        self._index: List[Tuple[str, int]] = []
        # Node count per event, so a batch can be budgeted by hits instead of by events.
        # It equals the hit count exactly, since this encoding emits one node per hit.
        self.sizes: List[int] = []
        for seed in range(lo, hi + 1):
            seed_dir = os.path.join(data_dir, f"seed_{seed}")
            if not os.path.isdir(seed_dir):
                continue
            counts: Dict[int, int] = {}
            found = False
            for pattern in ("dc_hits_*.parquet", "vtx_hits_*.parquet"):
                matches = glob.glob(os.path.join(seed_dir, pattern))
                if not matches:
                    continue
                found = True
                ids, n = np.unique(
                    pq.read_table(matches[0], columns=["event_id"])
                    .column("event_id").to_numpy(), return_counts=True)
                for e, c in zip(ids.tolist(), n.tolist()):
                    counts[int(e)] = counts.get(int(e), 0) + int(c)
            if not found:
                continue
            ordered = sorted(counts)
            if max_events_per_seed is not None:
                ordered = ordered[:max_events_per_seed]
            self._index.extend((seed_dir, e) for e in ordered)
            self.sizes.extend(counts[e] for e in ordered)

        if not self._index:
            raise RuntimeError(
                f"no events found under {data_dir} for seeds {lo}-{hi}"
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        seed_dir, event_id = self._index[i]
        dc_table, vtx_table, mc_table = _read_seed(seed_dir)
        dc = dc_table.event(event_id)
        vtx = vtx_table.event(event_id)
        mc = mc_table.event(event_id)

        def col(event, name, dtype=torch.float32):
            return torch.as_tensor(np.ascontiguousarray(event[name]), dtype=dtype)

        # Vertex hits sit at their own position and carry no displacement; drift hits sit at
        # the left tangency point and carry right - left. This is their vector=True branch.
        vtx_pos = torch.stack(
            [col(vtx, "hit_x"), col(vtx, "hit_y"), col(vtx, "hit_z")], dim=1)
        dc_left = torch.stack(
            [col(dc, "left_x"), col(dc, "left_y"), col(dc, "left_z")], dim=1)
        dc_right = torch.stack(
            [col(dc, "right_x"), col(dc, "right_y"), col(dc, "right_z")], dim=1)

        pos = torch.cat([vtx_pos, dc_left], dim=0)
        vector = torch.cat([torch.zeros_like(vtx_pos), dc_right - dc_left], dim=0)
        hit_type = torch.cat([col(vtx, "hit_type"), col(dc, "hit_type")], dim=0)
        link_original = torch.cat(
            [col(vtx, "mc_index", torch.long), col(dc, "mc_index", torch.long)], dim=0)
        is_secondary = torch.cat(
            [col(vtx, "produced_by_secondary", torch.long),
             col(dc, "produced_by_secondary", torch.long)], dim=0)

        link = (apply_garbage_label(link_original, is_secondary, self.min_num_hits)
                if self.garbage_label else link_original.clone())
        cluster_id = find_cluster_id(link)

        n = pos.shape[0]
        g = dgl.graph(([], []), num_nodes=n)
        g.ndata["pos_hits_xyz"] = pos
        g.ndata["vector"] = vector
        g.ndata["hit_type"] = hit_type
        g.ndata["particle_number"] = cluster_id.to(torch.int64)
        g.ndata["particle_number_nomap"] = link
        g.ndata["particle_number_nomap_original"] = link_original
        g.ndata["isSecondary"] = is_secondary.view(-1, 1)
        # Carried because their graph has them. We have no cell ids or overlay in this
        # dataset, and nothing downstream reads either, so they are constant.
        g.ndata["cellid"] = torch.zeros(n, 1)
        g.ndata["is_overlay"] = torch.zeros(n)
        g.ndata["eventNumber"] = torch.full((n,), event_id, dtype=torch.long)
        g.ndata["fileNumber"] = torch.zeros(n, dtype=torch.long)

        # Truth: one row per surviving cluster, in cluster-id order, so row j corresponds to
        # cluster j+1. Their loss builds an object-indexed tensor the same way, so any other
        # ordering silently pairs a particle's angles with a different object.
        signal_links = torch.unique(link[link != -1])
        mc_lookup = {int(m): r for r, m in enumerate(mc["mc_index"])}
        missing = [int(l) for l in signal_links.tolist() if int(l) not in mc_lookup]
        if missing:
            raise KeyError(
                f"event {event_id} in {seed_dir}: hits reference particles absent from the "
                f"truth table ({missing[:5]}). Their loss indexes objects by cluster id, so a "
                f"gap here would pair a cluster with another particle's angles."
            )
        rows = np.array([mc_lookup[int(l)] for l in signal_links.tolist()], dtype=np.int64)
        y = torch.zeros(len(rows), Y_NCOLS, dtype=torch.float32)
        if len(rows):
            sel = {c: a[rows] for c, a in mc.items()}
            y[:, Y_THETA] = col(sel, "theta")
            y[:, Y_PHI] = col(sel, "phi")
            y[:, Y_MASS] = col(sel, "mass")
            y[:, Y_PDG] = col(sel, "pdg")
            y[:, Y_PART_ID] = col(sel, "mc_index")
            y[:, Y_P] = col(sel, "p")
            y[:, Y_PT] = col(sel, "pt")
            y[:, Y_GEN_STATUS] = col(sel, "gen_status")
            y[:, Y_PARENT] = col(sel, "parent_index")
        # The batch index is appended by the collate, the only place it is known.
        return g, y


class TokenBudgetEventSampler(Sampler):
    """Batch events up to a hit budget rather than to a fixed event count.

    Their recipe batches 8 events, which is safe for their event sizes and not for ours: their
    example event has 694 nodes, ours run 912 to 10540 with a median of 3378. Eight of ours is
    about five times the tokens of eight of theirs, and because the spread is tenfold, a fixed
    event count makes the batch's memory footprint vary by the same factor -- measured, as an
    out-of-memory failure on an unlucky draw at a batch size that had just succeeded.

    Budgeting on hits fixes that and has two further benefits the comparison needs: both arms
    see identical batches, and an epoch is a predictable number of optimizer steps.

    Epoch length is pinned for the reason documented at length in our own sampler: Lightning
    derives its end-of-epoch validation trigger from the first epoch's batch count, so an epoch
    that packs even one batch shorter is never validated.

    **DDP-aware.** Every rank builds the identical global batch list (the shuffle is seeded
    by epoch only), truncates it to a multiple of world_size so all ranks step in lockstep
    (no one-rank-finishes-first NCCL deadlocks), and takes the strided slice
    `batches[rank::world_size]` -- each batch lands on exactly one rank. Ported from our own
    `TokenBudgetBatchSampler`, which has run 4-GPU DDP all campaign. With WORLD_SIZE unset or
    1 this is a no-op and the original single-process behavior is unchanged.
    """

    def __init__(self, sizes: List[int], max_tokens: int, shuffle: bool = True,
                 stable_epoch_length: bool = True, probe_epochs: int = 64,
                 verbose: bool = True):
        self.sizes = list(sizes)
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.stable_epoch_length = stable_epoch_length
        self.probe_epochs = probe_epochs
        self.verbose = verbose
        self._epoch = 0
        self._fixed: Optional[int] = None
        self._cached: Optional[List[List[int]]] = None

    def _pack(self) -> List[List[int]]:
        order = list(range(len(self.sizes)))
        # Sort by size first so a batch holds events of similar length, then shuffle in blocks,
        # which keeps the padding waste down without making the order deterministic.
        order.sort(key=lambda i: self.sizes[i])
        if self.shuffle:
            rng = random.Random(42 + self._epoch)
            block = 80
            blocks = [order[i:i + block] for i in range(0, len(order), block)]
            rng.shuffle(blocks)
            for b in blocks:
                rng.shuffle(b)
            order = [i for b in blocks for i in b]

        batches, cur, tokens = [], [], 0
        for i in order:
            if cur and tokens + self.sizes[i] > self.max_tokens:
                batches.append(cur)
                cur, tokens = [], 0
            cur.append(i)
            tokens += self.sizes[i]
        if cur:
            batches.append(cur)
        if self.shuffle:
            random.Random(1000 + self._epoch).shuffle(batches)
        return batches

    def _target(self) -> int:
        if self._fixed is None:
            saved, counts = self._epoch, []
            for e in range(max(self.probe_epochs, 1)):
                self._epoch = e
                counts.append(len(self._pack()))
            self._epoch = saved
            self._fixed = min(counts)
            if self.verbose:
                print(f"  TokenBudgetEventSampler: {self._fixed} batches/epoch at "
                      f"max_tokens={self.max_tokens} (probe spanned "
                      f"{min(counts)}-{max(counts)})", flush=True)
        return self._fixed

    @staticmethod
    def _rank_info():
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        return rank, world_size

    def _build(self) -> List[List[int]]:
        batches = self._pack()
        if self.stable_epoch_length and self.shuffle:
            batches = batches[:self._target()]
        rank, world = self._rank_info()
        if world > 1:
            usable = len(batches) - (len(batches) % world)
            batches = batches[rank:usable:world]
        return batches

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self._cached = self._build()

    def __iter__(self):
        if self._cached is None:
            self._cached = self._build()
        yield from self._cached

    def __len__(self) -> int:
        if self._cached is None:
            self._cached = self._build()
        return len(self._cached)


def collate_ggtf(samples) -> Tuple[dgl.DGLGraph, torch.Tensor]:
    """Batch graphs and append each truth row's batch-local event index.

    Mirrors their `add_batch_number` in `layers/batch_operations.py`, including that the index
    is appended rather than written into a reserved slot. Their loss splits the table by
    `y[:, -1] == i` for i over `dgl.unbatch(g)`, so it must be the position in this batch and
    not the event id on disk.
    """
    graphs, ys = zip(*samples)
    stamped = [
        torch.cat((y, torch.full((y.shape[0], 1), float(i), dtype=y.dtype)), dim=1)
        for i, y in enumerate(ys)
    ]
    return dgl.batch(graphs), torch.cat(stamped, dim=0)


def _self_test(data_dir: str, seeds: str) -> None:
    ds = ParquetGGTFDataset(data_dir, parse_seed_range(seeds), max_events_per_seed=4)
    print(f"{len(ds)} events indexed")

    g, y = ds[0]
    n_dc = int((g.ndata["hit_type"] == 0).sum())
    n_vtx = int((g.ndata["hit_type"] == 1).sum())
    n_noise = int((g.ndata["particle_number"] == 0).sum())
    print(f"\nevent 0: {g.num_nodes()} nodes ({n_vtx} vertex, {n_dc} drift), "
          f"{g.num_edges()} edges")
    print(f"  clusters: {int(g.ndata['particle_number'].max())} signal, "
          f"{n_noise} nodes relabelled noise")
    print(f"  truth rows: {y.shape[0]}  (must equal the signal cluster count)")
    assert y.shape[0] == int(g.ndata["particle_number"].max()), "y/cluster mismatch"

    # Vertex hits must carry exactly no displacement, drift hits must carry 2*drift_distance.
    vtx_mask = g.ndata["hit_type"] == 1
    assert torch.all(g.ndata["vector"][vtx_mask] == 0), "vertex hits carry a displacement"
    dc_norms = g.ndata["vector"][~vtx_mask].norm(dim=1)
    print(f"  drift displacement |right-left|: mean {dc_norms.mean():.3f} mm, "
          f"max {dc_norms.max():.3f} mm")

    # Cluster ids must be dense 1..K with no gaps, or the loss indexes a nonexistent object.
    ids = torch.unique(g.ndata["particle_number"])
    expected = torch.arange(0, int(ids.max()) + 1)
    assert torch.equal(ids, expected), f"cluster ids not dense: {ids[:20]}"
    print(f"  cluster ids dense 0..{int(ids.max())} (0 = noise)")

    # The relabelling must keep every hit. This is the property that distinguishes it from
    # our --drop_loopers, so it is asserted rather than trusted.
    ds_raw = ParquetGGTFDataset(data_dir, parse_seed_range(seeds),
                               garbage_label=False, max_events_per_seed=4)
    g_raw, y_raw = ds_raw[0]
    assert g_raw.num_nodes() == g.num_nodes(), "relabelling changed the node count"
    print(f"\nrelabelling kept all {g.num_nodes()} nodes and moved "
          f"{y_raw.shape[0] - y.shape[0]} particles out of the target set "
          f"({y_raw.shape[0]} -> {y.shape[0]})")

    bg, by = collate_ggtf([ds[0], ds[1], ds[2]])
    print(f"\nbatched: {bg.num_nodes()} nodes over {len(bg.batch_num_nodes())} events, "
          f"truth {tuple(by.shape)} ({Y_NCOLS} of theirs + appended batch index)")
    assert by.shape[1] == Y_NCOLS + 1, "batch index not appended"
    print(f"  event index column values: {sorted(set(by[:, Y_EVENT_IDX].tolist()))}")
    for i in range(len(bg.batch_num_nodes())):
        rows = int((by[:, Y_EVENT_IDX] == i).sum())
        clusters = int(dgl.unbatch(bg)[i].ndata["particle_number"].max())
        assert rows == clusters, f"event {i}: {rows} truth rows vs {clusters} clusters"
    print("  every event's truth rows match its cluster count")
    print("\nOK: the adapter satisfies GGTF's graph and truth contract")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--seeds", default="1-2")
    a = ap.parse_args()
    _self_test(a.data_dir, a.seeds)
