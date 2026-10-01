"""Polars-based Parquet dataset for IDEA detector CGA track finding.

Lazy-loading: only stores lightweight metadata at init.
Tensors are built on-demand in __getitem__ with an LRU cache for parquet reads.
"""

import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, Sampler

try:
    import polars as pl
except ImportError:
    raise ImportError("Install polars: pip install polars")


def _read_and_group(parquet_path: str):
    """Read a parquet file and return a dict of {event_id: DataFrame}."""
    df = pl.read_parquet(parquet_path)
    grouped = {}
    for part in df.partition_by("event_id"):
        grouped[part["event_id"][0]] = part
    return grouped


# Module-level LRU cache — shared across dataset instances within one process.
# Keyed by file path string; caches the grouped dict so repeated __getitem__
# calls for events in the same seed don't re-read parquet.
_PARQUET_CACHE_SIZE = int(os.environ.get("CGATR_PARQUET_CACHE_SIZE", "64"))

@lru_cache(maxsize=_PARQUET_CACHE_SIZE)
def _cached_read(parquet_path: str):
    return _read_and_group(parquet_path)


@lru_cache(maxsize=_PARQUET_CACHE_SIZE)
def _cached_read_mc(parquet_path: str):
    """Grouped MC particle table, only read when daughter merging is on."""
    if not os.path.exists(parquet_path):
        return {}
    return _read_and_group(parquet_path)


def _normalize_time(t):
    """Compress hit time onto an O(1) range.

    Drift-chamber times span 1-450 ns, but vertex times reach -3.9 us to 192 us
    on backscatter and late decays, so a linear scale would bury the drift
    signal. A signed log keeps the 1-450 ns band well resolved while bounding
    the tails; the clamp only bites on the far outliers.
    """
    return (torch.sign(t) * torch.log1p(t.abs()) / 5.0).clamp_(-2.0, 2.5)


# Feature column where the drift-direction triple starts when there is no time
# column; add one when there is. The model derives the same offset from its own
# use_time flag, so the two must stay in step.
DRIFT_DIR_OFFSET = 10

# GGTF's remove_loopers thresholds, in mm, from their
# src/dataset/functions_graph_tracking.py. Their coordinate is the DC left
# tangency point, which our parquet carries directly as left_x/y/z.
_LOOPER_EXTENT_MM = (1600.0, 1600.0, 2800.0)
_LOOPER_MIN_HITS = 5

# fix_splitted_tracks reassigns a daughter's hits to its parent when the parent
# also has hits and the momentum in their y[:, 5] column exceeds this.
_DAUGHTER_MERGE_P_GEV = 0.01


def looper_mc_indices(dc_df, vtx_df):
    """The mc_index values GGTF's remove_loopers would delete from the graph.

    Their predicate: drop a particle when the bounding box of its hits spans
    more than 1600 mm in x or y or 2800 mm in z, or when it has fewer than 5
    hits. Both the hits and the particle disappear, so the network never sees
    them and they are not targets.

    Note what this is: a filter that needs the hit-to-particle association, so
    it cannot be applied to real data, and their own production
    GGTFTrackFinder in k4RecTracker receives every hit. Reproducing it lets us
    compare on their terms; it is not something we would deploy.

    Noise (mc_index <= 0) is never considered or removed. Note that this spares
    particle 0 from the looper predicate, which is the one place the mistaken
    noise convention happens to do no harm -- index 0 is a real particle, not
    noise (FINDINGS.md M20).
    """
    parts = []
    if len(dc_df):
        parts.append(dc_df.select(
            pl.col("mc_index"),
            pl.col("left_x").alias("x"),
            pl.col("left_y").alias("y"),
            pl.col("left_z").alias("z"),
        ))
    if len(vtx_df):
        parts.append(vtx_df.select(
            pl.col("mc_index"),
            pl.col("hit_x").alias("x"),
            pl.col("hit_y").alias("y"),
            pl.col("hit_z").alias("z"),
        ))
    if not parts:
        return set()

    agg = (
        pl.concat(parts)
        .filter(pl.col("mc_index") > 0)
        .group_by("mc_index")
        .agg(
            (pl.col("x").max() - pl.col("x").min()).alias("ex"),
            (pl.col("y").max() - pl.col("y").min()).alias("ey"),
            (pl.col("z").max() - pl.col("z").min()).alias("ez"),
            pl.len().alias("n"),
        )
    )
    lim_x, lim_y, lim_z = _LOOPER_EXTENT_MM
    bad = agg.filter(
        (pl.col("ex") > lim_x)
        | (pl.col("ey") > lim_y)
        | (pl.col("ez") > lim_z)
        | (pl.col("n") < _LOOPER_MIN_HITS)
    )
    return set(bad["mc_index"].to_list())


def drop_looper_particles(dc_df, vtx_df):
    """Remove the hits of every particle remove_loopers would delete."""
    bad = looper_mc_indices(dc_df, vtx_df)
    if not bad:
        return dc_df, vtx_df
    bad_list = list(bad)
    keep = ~pl.col("mc_index").is_in(bad_list)
    return dc_df.filter(keep), vtx_df.filter(keep)


def merge_daughter_particles(dc_df, vtx_df, mc_df):
    """GGTF's fix_splitted_tracks: fold a daughter's hits into its parent.

    Applied when the parent also left hits and the daughter's momentum exceeds
    0.01 GeV. Their loop reads `y[indx, 5]` where `indx` enumerates the daughter
    rows, so the momentum cut is on the daughter; the parent only has to exist
    in the hit list. Relations are resolved transitively so a chain of splits
    collapses onto one ancestor.
    """
    if mc_df is None or not len(mc_df):
        return dc_df, vtx_df
    with_hits = set()
    if len(dc_df):
        with_hits |= set(dc_df["mc_index"].to_list())
    if len(vtx_df):
        with_hits |= set(vtx_df["mc_index"].to_list())
    with_hits.discard(0)
    if not with_hits:
        return dc_df, vtx_df

    rel = mc_df.filter(
        pl.col("mc_index").is_in(list(with_hits))
        & (pl.col("p") > _DAUGHTER_MERGE_P_GEV)
        & pl.col("parent_index").is_in(list(with_hits))
    ).select(["mc_index", "parent_index"])
    if not len(rel):
        return dc_df, vtx_df

    parent_of = dict(zip(rel["mc_index"].to_list(), rel["parent_index"].to_list()))

    def ancestor(i):
        seen = set()
        while i in parent_of and i not in seen:
            seen.add(i)
            i = parent_of[i]
        return i

    remap = {i: ancestor(i) for i in parent_of}
    remap = {k: v for k, v in remap.items() if k != v}
    if not remap:
        return dc_df, vtx_df

    expr = pl.col("mc_index").replace(remap)
    return (dc_df.with_columns(expr) if len(dc_df) else dc_df,
            vtx_df.with_columns(expr) if len(vtx_df) else vtx_df)


def ggtf_track_separation_weights(
    dc_df, vtx_df, mc_df, min_target_hits: int
):
    """Per-hit inverse nearest-track distance used by GGTF's repulsive loss.

    GGTF computes Euclidean distance in truth (eta, phi), without wrapping phi,
    then weights each target by ``1 / (nearest_distance + 0.001)``. Only
    non-secondary particles that remain targets after the hit-count rule enter
    the nearest-neighbour calculation.
    """
    if mc_df is None:
        raise ValueError("GGTF object-condensation mode requires mc_particles")

    hit_truth = pl.concat([
        vtx_df.select(["mc_index", "produced_by_secondary"]),
        dc_df.select(["mc_index", "produced_by_secondary"]),
    ])
    target_ids = (
        hit_truth
        .filter(pl.col("produced_by_secondary") == 0)
        .group_by("mc_index")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= min_target_hits)["mc_index"]
        .to_list()
    )
    particles = (
        mc_df.filter(pl.col("mc_index").is_in(target_ids))
        .sort("mc_index")
        .select(["mc_index", "theta", "phi"])
    )
    if not len(particles):
        weight_by_id = {}
    elif len(particles) == 1:
        weight_by_id = {int(particles["mc_index"][0]): 1.0 / 1.001}
    else:
        theta = torch.tensor(particles["theta"].to_numpy(), dtype=torch.float32)
        phi = torch.tensor(particles["phi"].to_numpy(), dtype=torch.float32)
        eta = -torch.log(torch.tan(theta / 2))
        eta_phi = torch.stack([eta, phi], dim=1)
        distance = torch.cdist(eta_phi, eta_phi)
        nearest = distance.sort(dim=1).values[:, 1]
        weight = 1.0 / (nearest + 0.001)
        weight_by_id = dict(zip(
            particles["mc_index"].to_list(), weight.tolist()
        ))

    def hit_weights(df):
        return torch.tensor(
            [weight_by_id.get(int(i), 1.0) for i in df["mc_index"]],
            dtype=torch.float32,
        )

    return torch.cat([hit_weights(vtx_df), hit_weights(dc_df)], dim=0)


def gen_status_secondary_masks(dc_df, vtx_df, mc_df):
    """Per-hit secondary flag derived from the linked particle's `gen_status`.

    `produced_by_secondary` is identically zero in this production, and that is
    not a conversion bug (M83): EDM4hep sets `isProducedBySecondary` only when a
    hit's producing particle is *absent* from the MCParticle collection, whereas
    our simulation stores secondaries as their own MCParticles with
    `generatorStatus == 0` and links hits straight to them. The information is
    therefore in `gen_status`, more precisely than in the flag GGTF read.

    GGTF's `create_garbage_label` converts that population to noise, so matching
    their target definition on our data means deriving the flag here.
    """
    if mc_df is None:
        raise ValueError("secondaries_as_noise requires mc_particles")
    secondary_ids = mc_df.filter(pl.col("gen_status") == 0)["mc_index"]

    def mask_for(frame):
        if not len(frame):
            return torch.zeros(0, dtype=torch.bool)
        flags = frame["mc_index"].is_in(secondary_ids.implode()).to_numpy()
        return torch.tensor(flags, dtype=torch.bool)

    return mask_for(vtx_df), mask_for(dc_df)


def _build_event_tensors(dc_df, vtx_df, with_time: bool = False,
                         with_drift_dir: bool = False,
                         track_separation_weight=None,
                         secondary_masks=None):
    """Convert Polars DataFrames for one event into feature/label tensors.

    With `with_time`, an 11th feature column carries the normalized hit time.
    The column is opt-in because the evaluation scripts share this dataset and
    the trained checkpoints expect the 10-column layout.

    With `with_drift_dir`, three further columns carry the unit vector from the
    left to the right tangency point, i.e. the direction the ionisation drifted
    in. Only GGTF's projective hit encoding needs it, because that encoding
    places the hit at one tangency point and translates to the other. The
    columns sit after the time slot, so their offset depends on `with_time`.
    """
    n_dc = len(dc_df)
    n_vtx = len(vtx_df)
    n_total = n_dc + n_vtx

    vtx_pos = torch.tensor(
        vtx_df.select(["hit_x", "hit_y", "hit_z"]).to_numpy(), dtype=torch.float32
    )
    dc_wire = torch.tensor(
        dc_df.select([
            "wire_x", "wire_y", "wire_z",
            "drift_distance", "wire_azimuthal_angle", "wire_stereo_angle"
        ]).to_numpy(), dtype=torch.float32
    )
    dc_pos = torch.tensor(
        dc_df.select(["hit_x", "hit_y", "hit_z"]).to_numpy(), dtype=torch.float32
    )
    vtx_mc = torch.tensor(vtx_df["mc_index"].to_numpy(), dtype=torch.long)
    dc_mc = torch.tensor(dc_df["mc_index"].to_numpy(), dtype=torch.long)
    if secondary_masks is None:
        vtx_sec = torch.tensor(
            vtx_df["produced_by_secondary"].to_numpy(), dtype=torch.bool
        )
        dc_sec = torch.tensor(
            dc_df["produced_by_secondary"].to_numpy(), dtype=torch.bool
        )
    else:
        vtx_sec, dc_sec = secondary_masks

    n_cols = 10 + (1 if with_time else 0) + (3 if with_drift_dir else 0)
    features = torch.zeros(n_total, n_cols, dtype=torch.float32)
    features[:n_vtx, :3] = vtx_pos
    features[:n_vtx, 3] = 0.0
    features[n_vtx:, :3] = dc_pos
    features[n_vtx:, 3] = 1.0
    features[n_vtx:, 4:10] = dc_wire

    if with_time:
        vtx_t = torch.tensor(vtx_df["time"].to_numpy(), dtype=torch.float32)
        dc_t = torch.tensor(dc_df["time"].to_numpy(), dtype=torch.float32)
        features[:, 10] = _normalize_time(torch.cat([vtx_t, dc_t], dim=0))

    if with_drift_dir and n_dc:
        off = DRIFT_DIR_OFFSET + (1 if with_time else 0)
        lr = torch.tensor(
            dc_df.select(["left_x", "left_y", "left_z",
                          "right_x", "right_y", "right_z"]).to_numpy(),
            dtype=torch.float32)
        d = lr[:, 3:6] - lr[:, 0:3]
        # right - left is exactly 2 * drift_distance long and its midpoint is
        # exactly the wire, verified on 200k hits, so the unit vector is the
        # drift direction with no reconstruction needed. Vertex hits keep zeros.
        features[n_vtx:, off:off + 3] = d / (d.norm(dim=-1, keepdim=True) + 1e-8)

    event = {
        "features": features,
        "mc_index": torch.cat([vtx_mc, dc_mc], dim=0),
        "is_secondary": torch.cat([vtx_sec, dc_sec], dim=0),
        "n_hits": n_total,
        "n_vtx": n_vtx,
        "n_dc": n_dc,
    }
    if track_separation_weight is not None:
        event["track_separation_weight"] = track_separation_weight
    return event


class IDEAParquetDataset(Dataset):
    """Dataset that loads IDEA detector hits from Parquet files.

    Lazy-loading: __init__ only scans for valid (seed, event_id) pairs.
    Tensors are built on-demand in __getitem__.

    Expected directory structure:
        data_dir/
            seed_1/
                dc_hits_train.parquet
                vtx_hits_train.parquet
    """

    def __init__(
        self,
        data_dir: str,
        seed_range: Tuple[int, int] = None,
        seed_list=None,
        max_hits_per_event: Optional[int] = None,
        with_time: bool = False,
        drop_loopers: bool = False,
        merge_daughters: bool = False,
        with_drift_dir: bool = False,
        ggtf_loss: bool = False,
        min_target_hits: int = 3,
        secondaries_as_noise: bool = False,
    ):
        self.max_hits = max_hits_per_event
        self.with_time = with_time
        self.with_drift_dir = with_drift_dir
        self.ggtf_loss = ggtf_loss
        self.min_target_hits = min_target_hits
        # Match GGTF's target definition by deriving the secondary flag from
        # gen_status, since produced_by_secondary is empty here (M83). Off by
        # default: keeping secondaries as targets is the harder task and is what
        # every run up to the epoch-24 baseline trained against.
        self.secondaries_as_noise = secondaries_as_noise
        # GGTF-matched target definition. Off by default: our own results keep
        # curlers, which is the harder task and the one we claim credit for.
        #
        # The per-event sizes cached below are the UNFILTERED ones, so the token
        # sampler packs the same events into the same batches whether or not
        # loopers are dropped. That wastes some of the token budget but makes an
        # epoch mean the same number of optimizer steps in both regimes, which
        # is what the curler comparison needs -- otherwise removing loopers
        # shrinks events several-fold and confounds the target change with a
        # shorter schedule.
        self.drop_loopers = drop_loopers
        self.merge_daughters = merge_daughters
        # Store lightweight metadata: (dc_path, vtx_path, event_id, n_total)
        self._index = []

        data_dir = Path(data_dir)

        if seed_list is not None:
            seed_indices = seed_list
        elif seed_range is not None:
            seed_indices = list(range(seed_range[0], seed_range[1]))
        else:
            seed_indices = list(range(1, 601))

        for seed_idx in seed_indices:
            seed_dir = data_dir / f"seed_{seed_idx}"
            if not seed_dir.exists():
                continue

            dc_path = seed_dir / "dc_hits_train.parquet"
            vtx_path = seed_dir / "vtx_hits_train.parquet"

            if not dc_path.exists() or not vtx_path.exists():
                continue

            try:
                # Lightweight scan: only read event_id + row counts
                dc_counts = (
                    pl.scan_parquet(str(dc_path))
                    .group_by("event_id")
                    .agg(pl.len().alias("n"))
                    .collect()
                )
                vtx_counts = (
                    pl.scan_parquet(str(vtx_path))
                    .group_by("event_id")
                    .agg(pl.len().alias("n"))
                    .collect()
                )

                dc_map = dict(zip(
                    dc_counts["event_id"].to_list(),
                    dc_counts["n"].to_list(),
                ))
                vtx_map = dict(zip(
                    vtx_counts["event_id"].to_list(),
                    vtx_counts["n"].to_list(),
                ))

                common_ids = sorted(set(dc_map) & set(vtx_map))
                dc_str = str(dc_path)
                vtx_str = str(vtx_path)

                for eid in common_ids:
                    n_total = dc_map[eid] + vtx_map[eid]
                    # Store true size for sampler budgeting; truncation is in __getitem__
                    effective = min(n_total, max_hits_per_event) if max_hits_per_event else n_total
                    self._index.append((dc_str, vtx_str, eid, effective))

            except Exception as e:
                print(f"Warning: could not scan seed {seed_idx}: {e}")
                continue

        if self._index:
            sizes = [e[3] for e in self._index]
            sizes.sort()
            print(f"IDEAParquetDataset: {len(self._index)} events from {len(seed_indices)} seeds")
            print(f"  Hits/event: min={sizes[0]}, median={sizes[len(sizes)//2]}, "
                  f"max={sizes[-1]}, total={sum(sizes):,}")
        else:
            print(f"IDEAParquetDataset: 0 events from {len(seed_indices)} seeds")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        dc_path, vtx_path, eid, _ = self._index[idx]

        dc_grouped = _cached_read(dc_path)
        vtx_grouped = _cached_read(vtx_path)

        dc_df = dc_grouped.get(eid)
        vtx_df = vtx_grouped.get(eid)
        if dc_df is None or vtx_df is None:
            return None

        # GGTF's order: fix_splitted_tracks first, then remove_loopers, so a
        # daughter's hits count towards the parent's extent and hit total.
        mc_df = None
        if self.merge_daughters or self.ggtf_loss or self.secondaries_as_noise:
            mc_path = str(Path(dc_path).with_name("mc_particles_train.parquet"))
            mc_df = _cached_read_mc(mc_path).get(eid)
        if self.merge_daughters:
            dc_df, vtx_df = merge_daughter_particles(dc_df, vtx_df, mc_df)
        if self.drop_loopers:
            dc_df, vtx_df = drop_looper_particles(dc_df, vtx_df)
            if len(dc_df) + len(vtx_df) < 4:
                return None

        # Subsample oversized events to keep batches tractable.
        # NOTE: treat 0 (and None) as UNCAPPED. Previously `is not None` made
        # max_hits=0 truncate every event to 1 DC hit (max(0 - n_vtx, 1)).
        if self.max_hits:
            n_total = len(dc_df) + len(vtx_df)
            if n_total > self.max_hits:
                # Keep all VTX hits (few), subsample DC hits
                keep_dc = max(self.max_hits - len(vtx_df), 1)
                if keep_dc < len(dc_df):
                    dc_df = dc_df.sample(keep_dc, seed=idx)

        track_separation_weight = None
        if self.ggtf_loss:
            track_separation_weight = ggtf_track_separation_weights(
                dc_df, vtx_df, mc_df, self.min_target_hits
            )
        # Derived after any daughter merge or looper drop, so the mask lines up
        # with the frames actually returned.
        secondary_masks = None
        if self.secondaries_as_noise:
            secondary_masks = gen_status_secondary_masks(dc_df, vtx_df, mc_df)
        return _build_event_tensors(
            dc_df, vtx_df, with_time=self.with_time,
            with_drift_dir=self.with_drift_dir,
            track_separation_weight=track_separation_weight,
            secondary_masks=secondary_masks,
        )


def collate_idea_events(batch):
    """Custom collate: concatenates events and returns seq_lens for attention mask."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    features = torch.cat([b["features"] for b in batch], dim=0)
    mc_index = torch.cat([b["mc_index"] for b in batch], dim=0)
    is_secondary = torch.cat([b["is_secondary"] for b in batch], dim=0)
    seq_lens = [b["n_hits"] for b in batch]

    collated = {
        "features": features,
        "mc_index": mc_index,
        "is_secondary": is_secondary,
        "seq_lens": seq_lens,
    }
    if "track_separation_weight" in batch[0]:
        collated["track_separation_weight"] = torch.cat(
            [b["track_separation_weight"] for b in batch], dim=0
        )
    return collated


class TokenBudgetBatchSampler(Sampler):
    """Batch sampler that packs events up to a token (hit) budget.

    Uses pre-computed n_total from IDEAParquetDataset._index to form
    batches whose total hit count stays under max_tokens.

    DDP safety contract (v37 hardening):
      1. The batch list is built once per epoch in `set_epoch()` and cached, so
         `__len__` and `__iter__` agree byte-for-byte and the LR scheduler gets
         the correct step count.
      2. The global batch list is *truncated* to a multiple of world_size
         BEFORE the per-rank slice, so every rank yields the exact same
         number of batches. This eliminates the "one-rank-finishes-first"
         class of NCCL deadlocks.
      3. Events whose hit count exceeds `max_tokens` are admitted as singleton
         batches by default (they exceed the budget but at least train on the
         data). Pass `drop_oversized=True` to filter them, or raise
         `--max_tokens` to cover the full distribution.
      4. RNG seeded by epoch only — calling `__iter__` does NOT advance the
         seed, so DataLoader prefetching / multiple iterations within one epoch
         remain deterministic.
    """

    def __init__(self, dataset, max_tokens: int, shuffle: bool = True,
                 drop_last: bool = True, drop_oversized: bool = False,
                 verbose: bool = True, stable_epoch_length: bool = True,
                 probe_epochs: int = 64):
        self.sizes = [entry[3] for entry in dataset._index]
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.drop_oversized = drop_oversized
        self.verbose = verbose
        self.stable_epoch_length = stable_epoch_length
        self.probe_epochs = probe_epochs
        self._epoch = 0
        self._cached_batches = None
        self._fixed_global: Optional[int] = None

        # Filter oversized events once. Singletons exceeding max_tokens are the
        # main OOM risk under AMP + grad_checkpoint, and we cannot honour the
        # token budget for them anyway.
        if drop_oversized:
            n_oversized = sum(1 for s in self.sizes if s > max_tokens)
            if n_oversized > 0 and verbose:
                rank = int(os.environ.get("LOCAL_RANK", 0))
                if rank == 0:
                    largest = max(self.sizes)
                    print(
                        f"  TokenBudgetBatchSampler: dropping {n_oversized}/"
                        f"{len(self.sizes)} events with hits > max_tokens="
                        f"{max_tokens} (largest={largest}). Increase "
                        f"--max_tokens to keep them.",
                        flush=True,
                    )

    def _get_rank_info(self):
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        return rank, world_size

    def _stable_global_target(self, world_size: int) -> int:
        """One batch count for every epoch, so an epoch is a fixed number of steps.

        The packing is order-dependent, so reshuffling moves the total by a batch or two
        (measured: 1563-1565 global at max_tokens=22000 over the 60-seed train set). That
        wobble is not harmless. Lightning pins its end-of-epoch validation trigger to the
        *first* epoch's batch count and then fires on `(batch_idx + 1) % that == 0`, so any
        epoch that comes out even one batch short never reaches it and is not validated at
        all. That is exactly what happened to every 8-epoch run before this existed: epochs
        4 and 7 came out at 1563 against a pinned 1564 and were silently skipped.

        Pinning to the minimum over a probe of `probe_epochs` epochs costs about 0.1% of the
        steps in an epoch and buys both a real validation every epoch and an epoch that is
        genuinely a fixed number of optimizer steps.
        """
        if self._fixed_global is not None:
            return self._fixed_global

        saved = self._epoch
        counts = []
        for e in range(max(self.probe_epochs, 1)):
            self._epoch = e
            counts.append(len(self._pack_global_batches()))
        self._epoch = saved

        target = (min(counts) // world_size) * world_size
        self._fixed_global = target
        if self.verbose and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(
                f"  TokenBudgetBatchSampler: pinning every epoch to {target} global "
                f"batches ({target // world_size}/rank); probe over "
                f"{max(self.probe_epochs, 1)} epochs spanned "
                f"{min(counts)}-{max(counts)}",
                flush=True,
            )
        return target

    def _pack_global_batches(self):
        """The order-dependent part: the global batch list before any DDP truncation."""
        rng = random.Random(42 + self._epoch)

        indices = [
            i for i, s in enumerate(self.sizes)
            if (not self.drop_oversized) or s <= self.max_tokens
        ]
        # Sort by size so similar events group together (better GPU util).
        indices.sort(key=lambda i: self.sizes[i])

        if self.shuffle:
            bucket_size = 80
            buckets = [indices[i:i + bucket_size]
                       for i in range(0, len(indices), bucket_size)]
            rng.shuffle(buckets)
            for bucket in buckets:
                rng.shuffle(bucket)
            indices = [idx for bucket in buckets for idx in bucket]

        # Pack into batches respecting token budget.
        batches = []
        current_batch = []
        current_tokens = 0
        for idx in indices:
            event_size = self.sizes[idx]
            if current_batch and current_tokens + event_size > self.max_tokens:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(idx)
            current_tokens += event_size
        if current_batch:
            if not self.drop_last or len(batches) == 0:
                batches.append(current_batch)

        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def _build_batches(self):
        """Build the canonical (rank-sliced, equal-length) batch list for the
        current epoch. Called lazily by both __iter__ and __len__ so they
        agree."""
        rank, world_size = self._get_rank_info()
        batches = self._pack_global_batches()

        # DDP: truncate to multiple of world_size BEFORE per-rank slice so all
        # ranks yield the exact same number of batches. This is the key
        # invariant that prevents NCCL collective desync.
        n_keep = (len(batches) // world_size) * world_size
        if self.stable_epoch_length and self.shuffle:
            target = self._stable_global_target(world_size)
            if len(batches) >= target:
                n_keep = target
            elif self.verbose and int(os.environ.get("LOCAL_RANK", 0)) == 0:
                # Only reachable past the probe horizon, and it reopens the skipped-
                # validation hole for this epoch, so say so rather than drift quietly.
                print(
                    f"  TokenBudgetBatchSampler: epoch {self._epoch} packed into "
                    f"{len(batches)} batches, below the pinned {target}; this epoch is "
                    f"short and Lightning will not validate it. Raise probe_epochs.",
                    flush=True,
                )
        batches = batches[:n_keep]
        if world_size > 1:
            batches = batches[rank::world_size]
        return batches

    def set_epoch(self, epoch: int):
        """Set epoch and rebuild the cached batch list."""
        self._epoch = epoch
        self._cached_batches = self._build_batches()

    def __iter__(self):
        if self._cached_batches is None:
            self._cached_batches = self._build_batches()
        yield from self._cached_batches

    def __len__(self):
        if self._cached_batches is None:
            self._cached_batches = self._build_batches()
        return len(self._cached_batches)
