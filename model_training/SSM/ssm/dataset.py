"""Data loading + featurisation for the SSM track finder. Independent package.

Input data layout (same files CIRCE reads):

    <data_dir>/<seed>/dc_hits_<split>.parquet    # drift-chamber hits
    <data_dir>/<seed>/vtx_hits_<split>.parquet   # vertex-detector hits

Expected columns:
  dc : event_id, hit_x,y,z, wire_x,y,z, drift_distance, wire_azimuthal_angle,
       wire_stereo_angle, mc_index
  vtx: event_id, hit_x,y,z, mc_index

Canonical per-hit feature vector (12 dims, this module's contract):

  [0:3]  x, y, z            (mm / 1000)
  [3]    d_drift            (mm / 10; 0 for silicon)
  [4:6]  sin(az), cos(az)   wire azimuthal direction (0 for silicon)
  [6]    stereo_angle       wire stereo angle radians (0 for silicon)
  [7]    hit_type           0 = vertex/silicon, 1 = drift chamber
  [8]    r                  (mm / 1000)
  [9]    phi
  [10:12] u, v              conformal coords * 1000

Ordering: hits are sorted by (hit_type, phi) [vertex first, then chamber],
which are exactly items [7] and [9]. Sequences are padded AT THE END to a
multiple of chunk_size; causality (see model.py) keeps padding harmless.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

# Graphs parquets carry awkward-extension-typed columns; force storage loads
# (Polars >= 2.0 would otherwise load_as_extension and fail).
import os as _os
_os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import polars as pl

FEATURE_DIM = 12
COORD_SCALE = 1.0 / 1000.0      # mm -> m
DRIFT_SCALE = 1.0 / 10.0        # mm -> 10 mm units
CONFORMAL_SCALE = 1000.0        # 1/mm -> 1/m units
NOISE_MC_INDEX = 0

_DC_REQUIRED = ["event_id", "hit_x", "hit_y", "hit_z", "mc_index", "drift_distance",
                "wire_x", "wire_y", "wire_z", "wire_azimuthal_angle", "wire_stereo_angle"]
_VTX_REQUIRED = ["event_id", "hit_x", "hit_y", "hit_z", "mc_index"]


def featurize(hits: dict) -> dict:
    """Build canonical 12-dim features + derived geometry keys from raw tensors.

    hits: dict with keys x,y,z (N), drift_d (N), wire_az (N), wire_st (N),
          hit_type (N, 0 vtx | 1 dc), mc_index (N)
    Returns dict with features (N,12), phi (N), mc_index (N).
    """
    x, y, z = hits["x"] * COORD_SCALE, hits["y"] * COORD_SCALE, hits["z"] * COORD_SCALE
    r = torch.sqrt(x * x + y * y)
    phi = torch.atan2(y, x)
    rr_mm = hits["x"] ** 2 + hits["y"] ** 2
    u = CONFORMAL_SCALE * hits["x"] / rr_mm.clamp(min=1.0)
    v = CONFORMAL_SCALE * hits["y"] / rr_mm.clamp(min=1.0)

    feats = torch.stack([
        x, y, z,
        hits["drift_d"] * DRIFT_SCALE * hits["hit_type"],          # vtx: 0
        torch.sin(hits["wire_az"]) * hits["hit_type"],
        torch.cos(hits["wire_az"]) * hits["hit_type"],
        hits["wire_st"] * hits["hit_type"],
        hits["hit_type"],
        r, phi, u, v,
    ], dim=1)
    return {"features": feats.float(), "phi": phi, "mc_index": hits["mc_index"]}


def augment_phi(event: dict, delta: float) -> dict:
    """Rotate an event by delta around the beam axis (features cols 0,1,4,5,8,9,10,11)."""
    f = event["features"]
    c, s = np.cos(delta), np.sin(delta)
    x, y = f[:, 0].clone(), f[:, 1].clone()
    f[:, 0] = c * x - s * y
    f[:, 1] = s * x + c * y
    sn, cn = f[:, 4].clone(), f[:, 5].clone()
    f[:, 4] = c * sn - s * cn
    f[:, 5] = s * sn + c * cn
    f[:, 8] = torch.sqrt(f[:, 0] ** 2 + f[:, 1] ** 2)
    f[:, 9] = torch.atan2(f[:, 1], f[:, 0])
    f[:, 10] = f[:, 0] / (f[:, 8] ** 2).clamp(min=1e-6)
    f[:, 11] = f[:, 1] / (f[:, 8] ** 2).clamp(min=1e-6)
    event["phi"] = torch.atan2(f[:, 1], f[:, 0])
    return event


def _part1by2(n: torch.Tensor) -> torch.Tensor:
    """Spread 21 low bits of n with two zero bits between each (Morton helper)."""
    n = n & 0x1FFFFF
    n = (n | (n << 32)) & 0x1F00000000FFFF
    n = (n | (n << 16)) & 0x1F0000FF0000FF
    n = (n | (n << 8)) & 0x100F00F00F00F00F
    n = (n | (n << 4)) & 0x10C30C30C30C30C3
    n = (n | (n << 2)) & 0x1249249249249249
    return n


def morton_key(x_mm: torch.Tensor, y_mm: torch.Tensor, z_mm: torch.Tensor,
               extent_mm: float = 4000.0) -> torch.Tensor:
    """Deterministic space-filling-curve key: bit-interleave of quantized coords."""
    bits = 21
    span = 2 * extent_mm
    def q(v):
        return ((v + extent_mm) / span * ((1 << bits) - 1)).clamp(0, (1 << bits) - 1).long()
    return _part1by2(q(x_mm)) | (_part1by2(q(y_mm)) << 1) | (_part1by2(q(z_mm)) << 2)


ORDER_MODES = ("spatial", "type_phi")


def sort_event(event: dict, mode: str = "spatial") -> dict:
    """Order hits into the SSM input sequence.

    mode="spatial"  : (hit_type, morton(x,y,z)) — set-native 3D locality; the
                      default because it preserves locality for displaced
                      tracks and loopers/curlers, which a phi-sort breaks.
    mode="type_phi" : (hit_type, phi) — vertex first, chamber by azimuth;
                      best locality for clean primaries, kept as ablation.
    """
    f, phi = event["features"], event["phi"]
    if mode == "spatial":
        key = morton_key(f[:, 0] / COORD_SCALE, f[:, 1] / COORD_SCALE, f[:, 2] / COORD_SCALE)
    elif mode == "type_phi":
        key = ((phi + torch.pi) * (2.0 ** 40)).long()
    else:
        raise ValueError(f"unknown order mode '{mode}' (available: {ORDER_MODES})")
    order = torch.argsort(key, stable=True)
    key2 = f[order, 7]  # hit_type: vtx first within equal keys
    order = order[torch.argsort(key2, stable=True)]
    event["features"] = f[order]
    event["mc_index"] = event["mc_index"][order]
    event["phi"] = phi[order]
    return event


def _schema(path: str):
    return pl.read_parquet_schema(path)


def _detect_layout(files):
    """Detect parquet layout from the first file's schema.

    Returns ("flat_hits" | "event_table", schema) or raises with the schema in
    the message (for offline debugging).
    """
    sch = _schema(str(files[0]))
    if "event_id" in sch and "hit_x" in sch:
        return "flat_hits", sch
    if "hit_x" in sch:
        dt = sch["hit_x"]
        # event-table: one row per event, list-typed hit columns. Check the
        # dtype REPR too: awkward-extended parquet columns show up as
        # Extension('awkward', List(...)) depending on the polars version.
        if (isinstance(dt, pl.List) or getattr(dt, "inner", None) is not None
                or "List(" in str(dt)):
            return "event_table", sch
        raise RuntimeError(
            f"Unsupported flat hit schema in {files[0]}: {dict(sch)}")
    raise RuntimeError(
        f"Unrecognized parquet layout in {files[0]}: columns={dict(sch)}\n"
        "Run `python -m ssm.inspect_schema <folder>` and report the output."
    )


class EventParquetDataset(Dataset):
    """Layout-agnostic per-event dataset over parquet files.

    Supported layouts (auto-detected from schema, not file names):
      - flat_hits: one row per HIT with columns event_id, hit_x/y/z, mc_index,
        and chamber-info columns (drift_distance, wire_*) for chamber hits
        and x/y/z only for silicon rows. Files may live in any sub-folder and
        be named anything; vertex/chamber separation follows column PRESENCE
        per file (all-chamber file => same layout as dc_hits_*.parquet).
      - event_table: one row per EVENT with list-typed columns
        hit_x/y/z, leftPosition_x/y/z, rightPosition_x/y/z, hit_type,
        hit_particle_index (GATr Graphs_* convention). drift distance and
        wire angles are derived from the left/right tangency points
        (right-left has length 2*drift_distance, midpoint = wire position).

    --split filters file names containing '_<split>.'; if no file matches,
    ALL parquet files are used (no-split folders).
    """

    def __init__(self, data_dir: str, split: str | None = "train",
                 max_events_per_seed=None, order_mode: str = "spatial",
                 max_files: int | None = None, max_events: int | None = None):
        if order_mode not in ORDER_MODES:
            raise ValueError(f"unknown order mode '{order_mode}'")
        self.order_mode = order_mode
        self.data_dir = Path(data_dir)
        files = sorted(self.data_dir.rglob("*.parquet"))
        if not files:
            raise RuntimeError(f"no *.parquet files under {data_dir}")
        if split:
            tagged = [f for f in files if f"_{split}." in f.name]
            if tagged:
                files = tagged
        self.layout, _ = _detect_layout(files)
        kept, skipped = [], 0
        for f in files:
            lay, _ = _detect_layout([f])
            if lay == self.layout:
                kept.append(f)
            else:
                skipped += 1
        if skipped:
            print(f"EventParquetDataset: WARNING skipped {skipped} file(s) whose "
                  f"layout differs from the dominant '{self.layout}'", flush=True)
        files = kept
        if max_files:
            files = files[:max_files]
        self.files = [str(f) for f in files]
        self._index = []     # (file_idx, event_key)  event_key: int id (flat) or row (event_table)
        self._sizes = []     # hits per event
        for fi, fstr in enumerate(self.files):
            if self.layout == "flat_hits":
                counts = (pl.scan_parquet(fstr).group_by("event_id").len()
                          .collect().sort("event_id"))
                ids, ls = counts["event_id"].to_list(), counts["len"].to_list()
            else:
                lens = (pl.scan_parquet(fstr)
                        .select(pl.col("hit_x").list.len().alias("n")).collect()["n"].to_list())
                ids, ls = list(range(len(lens))), lens
            n_ev = 0
            for k, l in zip(ids, ls):
                if max_events is not None and len(self._index) >= max_events:
                    break
                self._index.append((fi, k))
                self._sizes.append(int(l))
                n_ev += 1
                if max_events_per_seed is not None and n_ev >= max_events_per_seed:
                    break
            if max_events is not None and len(self._index) >= max_events:
                break
        if not self._index:
            raise RuntimeError(f"0 events from {len(files)} files under {data_dir}")
        print(f"EventParquetDataset: layout={self.layout}, files={len(files)}, "
              f"events={len(self._index)}, dir={data_dir}", flush=True)
        self._file_cache: dict = {}

    def _load_flat_grouped(self, path):
        if path not in self._file_cache:
            df = pl.read_parquet(path)
            if len(self._file_cache) > 8:
                self._file_cache.clear()
            self._file_cache[path] = {int(p["event_id"][0]): p
                                      for p in df.partition_by("event_id")}
        return self._file_cache[path]

    def _load_event_table(self, path):
        if path not in self._file_cache:
            if len(self._file_cache) > 8:
                self._file_cache.clear()
            self._file_cache[path] = pl.read_parquet(path)
        return self._file_cache[path]

    def __len__(self):
        return len(self._index)

    def event_size(self, i):
        return self._sizes[i]

    def __getitem__(self, i):
        fi, key = self._index[i]
        if self.layout == "flat_hits":
            df = self._load_flat_grouped(self.files[fi]).get(key)
            if df is None or len(df) == 0:
                raise KeyError(f"event {key} not found in {self.files[fi]}")
            x = torch.tensor(df["hit_x"].to_numpy(), dtype=torch.float32)
            y = torch.tensor(df["hit_y"].to_numpy(), dtype=torch.float32)
            z = torch.tensor(df["hit_z"].to_numpy(), dtype=torch.float32)
            n = len(df)
            cols = set(df.columns)
            if "drift_distance" in cols:
                dr = torch.tensor(df["drift_distance"].fill_null(0.0).to_numpy(), dtype=torch.float32)
                az = torch.tensor(df["wire_azimuthal_angle"].fill_null(0.0).to_numpy(), dtype=torch.float32)
                st = torch.tensor(df["wire_stereo_angle"].fill_null(0.0).to_numpy(), dtype=torch.float32)
                # per-hit type: chamber iff wire info exists for the row
                if "wire_x" in cols:
                    ht = torch.tensor(df["wire_x"].is_not_null().to_numpy(), dtype=torch.float32)
                elif "hit_type" in cols:
                    ht = torch.tensor(df["hit_type"].to_numpy(), dtype=torch.float32)
                else:
                    ht = torch.ones(n)
            else:
                dr = torch.zeros(n); az = torch.zeros(n); st = torch.zeros(n)
                ht = torch.tensor(df["hit_type"].to_numpy(), dtype=torch.float32) \
                    if "hit_type" in cols else torch.zeros(n)
            mc_col = "mc_index" if "mc_index" in cols else "hit_particle_index"
            mc = torch.tensor(df[mc_col].to_numpy(), dtype=torch.long)
            pt_map = None
            disp_map = None
            theta_map = None
            genstatus_map = None
        else:
            df = self._load_event_table(self.files[fi])
            row = df.row(key, named=True)
            x = torch.tensor(row["hit_x"], dtype=torch.float32)
            y = torch.tensor(row["hit_y"], dtype=torch.float32)
            z = torch.tensor(row["hit_z"], dtype=torch.float32)
            n = len(row["hit_x"])
            if "hit_type" in df.columns:
                ht_file = torch.tensor(row["hit_type"], dtype=torch.float32)
            else:
                ht_file = torch.zeros(n)
            # Real GATr/Graphs convention (verified on Zuds data): file
            # hit_type==0 = drift chamber, ==1 = silicon/VTX. Our internal
            # contract: 0 = vtx, 1 = chamber. Explicit drift/wire columns are
            # preferred over deriving from left/right tangency points.
            if "drift_distance" in df.columns:
                dc = (ht_file < 0.5).float()
                dr = torch.tensor(row["drift_distance"], dtype=torch.float32) * dc
                az = (torch.tensor(row["wire_azimuthal_angle"], dtype=torch.float32)
                      if "wire_azimuthal_angle" in df.columns else torch.zeros(n)) * dc
                st = (torch.tensor(row["wire_stereo_angle"], dtype=torch.float32)
                      if "wire_stereo_angle" in df.columns else torch.zeros(n)) * dc
                ht = dc
                lx = None
            elif "leftPosition_x" in df.columns:
                ht = ht_file
                lx = torch.tensor(row["leftPosition_x"], dtype=torch.float32)
                ly = torch.tensor(row["leftPosition_y"], dtype=torch.float32)
                lz = torch.tensor(row["leftPosition_z"], dtype=torch.float32)
                rx = torch.tensor(row["rightPosition_x"], dtype=torch.float32)
                ry = torch.tensor(row["rightPosition_y"], dtype=torch.float32)
                rz = torch.tensor(row["rightPosition_z"], dtype=torch.float32)
            elif "left_x" in df.columns:
                lx = torch.tensor(row["left_x"], dtype=torch.float32)
                ly = torch.tensor(row["left_y"], dtype=torch.float32)
                lz = torch.tensor(row["left_z"], dtype=torch.float32)
                rx = torch.tensor(row["right_x"], dtype=torch.float32)
                ry = torch.tensor(row["right_y"], dtype=torch.float32)
                rz = torch.tensor(row["right_z"], dtype=torch.float32)
            else:
                lx = ly = lz = rx = ry = rz = None
            if lx is not None:
                # derive drift/wire angles from left/right tangency points
                dx, dy, dz = rx - lx, ry - ly, rz - lz
                dist = torch.sqrt(dx * dx + dy * dy + dz * dz)
                dr = dist * 0.5 * ht
                az = torch.where(ht > 0.5, torch.atan2(dy, dx), torch.zeros(n))
                st = torch.where((ht > 0.5) & (dist > 1e-9),
                                 torch.asin((dz / dist).clamp(-1, 1)), torch.zeros(n))
            elif "drift_distance" not in df.columns:
                dr = torch.zeros(n); az = torch.zeros(n); st = torch.zeros(n)
            mc_col = "hit_particle_index" if "hit_particle_index" in df.columns else "mc_index"
            mc = torch.tensor(row[mc_col], dtype=torch.long)
            pt_map = None
            disp_map = None
            theta_map = None
            genstatus_map = None
            if "part_id" in df.columns and "part_p_t" in df.columns:
                pt_map = {int(p): float(t) for p, t in zip(row["part_id"], row["part_p_t"])}
                if "part_vertex_x" in df.columns:
                    disp_map = {
                        int(p): (float(x) ** 2 + float(y) ** 2) ** 0.5
                        for p, x, y in zip(row["part_id"], row["part_vertex_x"],
                                           row["part_vertex_y"])
                    }
                if "part_theta" in df.columns:
                    theta_map = {int(p): float(t) for p, t in
                                 zip(row["part_id"], row["part_theta"])}
                if "gen_status" in df.columns:
                    genstatus_map = {int(p): int(s) for p, s in
                                     zip(row["part_id"], row["gen_status"])}
        # noise convention: OC loss treats mc_index==0 as noise; shift any
        # negative noise label (GATr convention: -1) onto 0.
        if (mc < 0).any():
            mc = mc - mc.min()
        if (mc == 0).all() and n >= 8:
            raise ValueError(f"event {key}: ALL hits labeled noise (mc_index==0) "
                             "- check the label column")
        event = featurize({
            "x": x, "y": y, "z": z, "drift_d": dr,
            "wire_az": az, "wire_st": st, "hit_type": ht, "mc_index": mc,
        })
        event = sort_event(event, mode=self.order_mode)
        event["n_hits"] = n
        event["event_id"] = int(key)
        event["pt_map"] = pt_map if self.layout == "event_table" else None
        event["disp_map"] = disp_map if self.layout == "event_table" else None
        event["theta_map"] = theta_map if self.layout == "event_table" else None
        event["genstatus_map"] = genstatus_map if self.layout == "event_table" else None
        return event


class PackedBatchSampler(Sampler):
    """Pack events into batches of at most `token_budget` hits (post-padding).

    Events are size-bucket shuffled (CIRCE/GGTF global plan style), then packed
    greedily so batch lengths are uniform-ish.
    """

    def __init__(self, sizes, token_budget: int, chunk_size: int = 64, bucket: int = 50,
                 shuffle: bool = True, seed: int = 0, rank: int = 0, world_size: int = 1):
        self.budget = token_budget
        self.chunk = chunk_size
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        self.sizes = list(sizes)
        self.batches = []

    def _pad(self, n):
        return ((n + self.chunk - 1) // self.chunk) * self.chunk

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._cached_len = None

    def _pack(self, event_ids):
        packs, cur, cur_tokens = [], [], 0
        for i in event_ids:
            p = self._pad(self.sizes[i])
            if cur and cur_tokens + p > self.budget:
                packs.append(cur)
                cur, cur_tokens = [], 0
            cur.append(i)
            cur_tokens += p
        if cur:
            packs.append(cur)
        return packs

    def _build_shards(self):
        """LPT (longest-first) assign events to ranks -> per-rank greedy pack
        -> split-fill so every rank has the SAME number of batches.
        Zero duplication by construction: each event belongs to exactly one rank.
        """
        order = sorted(range(len(self.sizes)), key=lambda i: self.sizes[i], reverse=True)
        if self.shuffle:
            # same permutation on all ranks; re-seeded per epoch
            pre = order[:]
            random.Random(self.seed + self.epoch).shuffle(pre)
            # shuffle within coarse size buckets keeps packing tight-ish
            order = sorted(pre, key=lambda i: self.sizes[i], reverse=True)
        W = self.world_size
        shards = [[] for _ in range(W)]
        loads = [0] * W
        for i in order:
            r = loads.index(min(loads))
            shards[r].append(i)
            loads[r] += self._pad(self.sizes[i])
        packs = [self._pack(sh) for sh in shards]
        mx = max(len(p) for p in packs)
        # split-fill: sharing nothing, equalize counts by splitting longer packs
        for r in range(W):
            while len(packs[r]) < mx:
                big = max(packs[r], key=len)
                if len(big) < 2:
                    break
                mid = len(big) // 2
                packs[r].remove(big)
                packs[r].append(big[:mid])
                packs[r].append(big[mid:])
        self._n_batches_per_rank = mx
        return packs[self.rank]

    def __iter__(self):
        self.batches = self._build_shards()
        self._cached_len = len(self.batches)
        yield from self.batches

    def __len__(self):
        if not getattr(self, "_cached_len", None):
            # metadata-only build of the current-epoch shard to expose the count
            self._cached_len = len(self._build_shards())
        return self._cached_len


def collate_batch(events, chunk_size: int = 64):
    """Pad a list of events to a rectangular batch; padding appended AT THE END."""
    lens = [e["n_hits"] for e in events]
    N = max(lens)
    N = ((N + chunk_size - 1) // chunk_size) * chunk_size
    B = len(events)
    feats = torch.zeros(B, N, FEATURE_DIM)
    mc = torch.zeros(B, N, dtype=torch.long)
    for i, e in enumerate(events):
        n = e["n_hits"]
        feats[i, :n] = e["features"]
        mc[i, :n] = e["mc_index"]
        mc[i, n:] = NOISE_MC_INDEX            # padded hits are noise targets
    return {
        "features": feats,
        "mc_index": mc,
        "lengths": torch.tensor(lens, dtype=torch.long),
        "pt_maps": [e.get("pt_map") for e in events],
        "disp_maps": [e.get("disp_map") for e in events],
        "theta_maps": [e.get("theta_map") for e in events],
        "genstatus_maps": [e.get("genstatus_map") for e in events],
    }
