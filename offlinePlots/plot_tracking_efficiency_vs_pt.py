#!/usr/bin/env python3
"""Run tracking inference and plot tracking efficiency vs pT.

Two input modes, auto-detected from the parquet columns:

1.  Per-particle cache (columns ``pT``, ``theta``, ``matched`` or
    ``trackLabel`` ...): skips inference, applies cuts and plots directly.
2.  Hit-level Graphs parquet (columns ``part_p_t``, ``hit_x`` ...): requires
    ``--checkpoint``; loads the model exactly like
    ``model_training/GATR_CIRCE_LOSS/tests/evaluate_tracking_efficiency_pt.py``
    (EMA weights preferred, architecture reconstructed from checkpoint
    shapes), clusters the object-condensation output with the shared
    beta-ordered greedy clustering (``--tbeta``, ``--td``, ``--min-hits``),
    matches clusters to particles with the shared one-to-one assignment
    (``--matching-metric``), writes a per-particle cache parquet and plots.

Making the inference step reproducible: every inference run saves the
per-particle cache next to the plot (``<stem>_particles.parquet``); re-running
on that file skips the GPU step, so cuts/binning can be iterated cheaply.

Plot style mirrors ``trackingEfficiencyPlot`` in
``model_training/GATR/src/logger/plotting_tools.py``: log-spaced pT bins,
teal square markers, binomial errors clipped at 1, log-x major/minor locators,
dotted major/minor grids, and a textbox listing the truth selection.

Matching definition (``--matching-metric``): ``double_majority`` requires
hit efficiency >= 50% AND hit purity >= 50%; ``idea`` requires hit purity
>= 75% (GGTF definition 1). ``truth`` ids 0 are noise and never targets.

Examples
--------
# CPU inference + plot: 2 events per forward (dense-mask CPU attention
# allocates score tensors proportional to hits^2 -- watch RAM)
python plot_tracking_efficiency_vs_pt.py \
    /eos/.../Zuds/graph/Graphs_1_train.parquet \
    --checkpoint ckpts/model.ckpt --tbeta 0.35 --td 0.2 --batch-size 2 \
    --gen-status 0 1 --min-si-hits 2 --min-cdc-hits 5 -o eff.png

# or token-driven packing (training's TokenBudgetBatchSampler style)
python plot_tracking_efficiency_vs_pt.py \
    /eos/.../Zuds/graph/Graphs_1_train.parquet \
    --checkpoint ckpts/model.ckpt --max-batch-tokens 6000 -o eff.png

# iterate cuts on the saved cache, no GPU
python plot_tracking_efficiency_vs_pt.py eff_particles.parquet \
    --gen-status 1 --min-pt 0.5 -o eff_gen1.png
"""

import argparse
import glob as globmod
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
matplotlib.rc("font", size=15)
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MultipleLocator

_TEAL = "#238A8DFF"  # reference-plot marker/error colour

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_REPO = _SCRIPT_DIR.parent / "model_training" / "GATR_CIRCE_LOSS"
_EVAL_MODULE_NAME = "gatr_tracking_efficiency_eval"


def _natural_key(path):
    """Sort file paths naturally (Graphs_2 before Graphs_10)."""
    import re
    return [int(p) if p.isdigit() else p
            for p in re.split(r"(\d+)", str(path))]


def expand_input_files(values):
    """Expand positional inputs (files and/or glob patterns) to a sorted,
    de-duplicated file list."""
    files = []
    for value in values:
        hits = sorted(globmod.glob(value))
        files.extend(hits if hits else [value])
    seen, unique = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    unique.sort(key=_natural_key)
    return unique


# options that only have meaning when inference runs in this invocation;
# passing them with a per-particle cache input triggers a warning
_INFERENCE_ONLY_OPTS = (
    "checkpoint", "weights", "tbeta", "td", "min_hits",
    "rejected_seed_policy", "matching_metric", "max_files", "max_events",
    "device", "batch_size", "max_batch_tokens", "num_threads",
    "data_config", "layers_per_superlayer", "position_scale",
    "training_repo", "save_cache", "num_shards", "shard_id",
)


# ---------------------------------------------------------------------------
# import of the reference inference machinery (lazy: needs torch/dgl)
# ---------------------------------------------------------------------------
def _import_eval_module(repo: Path):
    """Load tests/evaluate_tracking_efficiency_pt.py from the training repo.

    The module hardcodes REPO_ROOT = its own parent directory (tests/), which
    only resolves correctly when the file is copied to the repo root; patch
    REPO_ROOT after import so gatr_v142 and config_files are found in place.
    """
    repo = repo.resolve()
    if not (repo / "src" / "gatr_v142").is_dir():
        raise SystemExit(
            f"--training-repo {repo} does not look like the GATR_CIRCE_LOSS "
            "tree (missing src/gatr_v142)"
        )
    eval_path = repo / "tests" / "evaluate_tracking_efficiency_pt.py"
    if not eval_path.is_file():
        raise SystemExit(f"reference evaluator not found: {eval_path}")

    for entry in (str(repo), str(repo.parent)):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    spec = importlib.util.spec_from_file_location(_EVAL_MODULE_NAME, eval_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPO_ROOT = repo
    return module


# ---------------------------------------------------------------------------
# inference: checkpoint -> per-event condensation output
# ---------------------------------------------------------------------------
def run_inference(args):
    """Return (events, meta) with per-event coords/beta/truth/hit_type/info."""
    import torch

    repo = Path(args.training_repo)
    ev = _import_eval_module(repo)

    checkpoint, state, hparams, weight_source = ev._load_checkpoint(
        args.checkpoint, args.weights
    )
    architecture = ev._architecture_from_state(state)
    from src.utils.detector_features import (
        DEFAULT_LAYERS_PER_SUPERLAYER,
        DETECTOR_FEATURE_NAMES,
        validate_layers_per_superlayer,
    )

    if architecture["detector_scalar_dim"] not in (0, len(DETECTOR_FEATURE_NAMES)):
        raise SystemExit(
            f"checkpoint expects {architecture['detector_scalar_dim']} detector "
            f"scalars; current code provides {len(DETECTOR_FEATURE_NAMES)}"
        )
    layers = validate_layers_per_superlayer(
        args.layers_per_superlayer
        if args.layers_per_superlayer is not None
        else hparams.get("layers_per_superlayer", DEFAULT_LAYERS_PER_SUPERLAYER)
    )
    position_scale = (
        args.position_scale
        if args.position_scale is not None
        else float(hparams.get("position_scale", 1000.0))
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        print(f"[inference] torch CPU threads: {args.num_threads}")

    print(
        f"[inference] checkpoint: {args.checkpoint}\n"
        f"[inference] weights: {weight_source}; "
        f"embedding_dim={architecture['embedding_dim']}, "
        f"blocks={architecture['num_blocks']}, "
        f"detector_scalar_dim={architecture['detector_scalar_dim']}, "
        f"helix_head={architecture['has_helix_head']}"
    )
    model = ev.CheckpointGATr(architecture, device=device,
                              position_scale=position_scale)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_required = [k for k in missing if not k.startswith("helix_proxy.")]
    unexpected_model = [
        k for k in unexpected
        if not k.startswith(("_ema.", "ema.", "criterion."))
    ]
    if missing_required or unexpected_model:
        raise SystemExit(
            f"checkpoint/model mismatch: missing={missing_required}, "
            f"unexpected={unexpected_model}"
        )
    model.to(device).eval()

    files = expand_input_files(args.parquet if isinstance(args.parquet, list)
                               else [args.parquet])
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise SystemExit(f"no input files match {args.parquet!r}")
    if args.max_files > 0:
        files = files[: args.max_files]
    fraction, available = ev._event_fraction(files, args.max_events)
    print(f"[inference] {len(files)} file(s), {available} events available; "
          f"reading first {fraction:.3%} of each file")

    data_config = args.data_config
    if data_config is None:
        config_name = (
            "config_tracking_parquet_detector.yaml"
            if architecture["detector_scalar_dim"]
            else "config_tracking_parquet.yaml"
        )
        data_config = str(repo / "config_files" / config_name)
    if args.num_shards > 1:
        # shard at row-group level: this process reads the shard's slice of
        # each file (load_range), so no event is decoded twice
        from src.dataset.dataset import SimpleIterDataset

        lo = args.shard_id / args.num_shards
        hi = (args.shard_id + 1) / args.num_shards
        dataset = SimpleIterDataset(
            {"evaluation": files},
            data_config,
            for_training=False,
            load_range_and_fraction=((lo, hi), fraction),
            fetch_by_files=True,
            fetch_step=1,
            async_load=False,
            infinity_mode=False,
            in_memory=False,
            name=f"tracking_efficiency_evaluation_shard{args.shard_id}",
            seed=42,
            dataset_cap=(args.max_events if args.max_events > 0 else None),
            layers_per_superlayer=tuple(layers),
        )
        print(f"[inference] shard {args.shard_id}/{args.num_shards}: "
              f"event range [{lo:.3f}, {hi:.3f}) of each file")
    else:
        dataset = ev._make_dataset(files, data_config, fraction,
                                   args.max_events, tuple(layers))

    # packing precedence: explicit --batch-size rules; --max-batch-tokens
    # alone switches to token-driven packing with unlimited events
    if args.batch_size is not None:
        if args.max_batch_tokens is not None:
            print("[inference] ignoring --max-batch-tokens because "
                  "--batch-size is set")
        batch_size = args.batch_size
        max_batch_tokens = 0
    elif args.max_batch_tokens is not None:
        batch_size = 1 << 30  # unlimited event cap; tokens drive packing
        max_batch_tokens = args.max_batch_tokens
    else:
        batch_size = 1
        max_batch_tokens = 0
    if device.type == "cpu" and batch_size > 1:
        print("[inference] note: dense-masked CPU attention allocates "
              "(heads, T, T) score tensors; memory scales as hits^2")
    events = _run_event_loop(ev, model, dataset, device, args.max_events,
                             batch_size, max_batch_tokens)
    meta = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "weights": weight_source,
        "embedding_dim": architecture["embedding_dim"],
        "files": files,
        "events": len(events),
        "device": str(device),
        "batch_size": (batch_size if batch_size < (1 << 30)
                       else "token-packed"),
        "max_batch_tokens": max_batch_tokens,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
    }
    return events, meta


def _forward_batched(model, batch_graph, device):
    """Forward over a dgl-batched graph with per-event attention blocks.

    Mirrors ``CheckpointGATr.forward`` of the reference evaluator, replacing
    the single-event assumptions (``attention_mask=None``, whole-graph
    ``join_reference``) with the batched equivalents used by the training
    wrapper (``ExampleWrapper`` in src/models/Gatr_withModifications.py):
    an xformers ``BlockDiagonalMask`` over per-event node counts and one
    data-derived join reference per event. Mathematically identical to
    running ``CheckpointGATr`` on each graph separately.
    """
    import torch

    dtype = model.clustering.weight.dtype
    position = batch_graph.ndata["pos_hits_xyz"].to(dtype=dtype)
    vector = batch_graph.ndata["vector"].to(dtype=dtype)
    hit_type = batch_graph.ndata["hit_type"].reshape(-1, 1).to(dtype=dtype)
    position = position / model.position_scale
    vector = vector / model.position_scale
    multivectors = (
        model._embed_point(position)
        + model._embed_scalar(hit_type)
        + model._embed_translation(vector)
    ).unsqueeze(-2)
    scalars = None
    if model.detector_scalar_dim:
        if "scalar_features" not in batch_graph.ndata:
            raise SystemExit(
                "detector-feature checkpoint requires "
                "graph.ndata['scalar_features']"
            )
        scalars = batch_graph.ndata["scalar_features"].to(dtype=dtype)

    seq_lens = [int(n) for n in batch_graph.batch_num_nodes()]
    if len(seq_lens) == 1:
        # identical to the reference single-event forward
        mask = None
    elif device.type == "cpu":
        # xformers' memory_efficient_attention has no CPU operator in this
        # build, but the fork dispatches AttentionBias instances to xformers
        # and plain Tensors to torch's scaled_dot_product_attention
        # (gatr/primitives/attention.py:460). So on CPU the block structure
        # must be a dense bool torch mask (True = attend). Beware its cost:
        # the CPU math backend materializes (heads, T, T) score tensors, T =
        # total batched hits; --max-batch-tokens keeps this bounded.
        total = sum(seq_lens)
        mask = torch.zeros(1, 1, total, total, dtype=torch.bool,
                           device=device)
        start = 0
        for length in seq_lens:
            mask[..., start : start + length, start : start + length] = True
            start += length
    else:
        from xformers.ops.fmha import BlockDiagonalMask

        # note: old xformers (as used by the training stack) has no `device`
        # kwarg; seqstart lives on CPU and the attention op handles it
        mask = BlockDiagonalMask.from_seqlens(seq_lens)
    join_references = []
    start = 0
    for length in seq_lens:
        sequence_mean = multivectors[start : start + length].mean(
            dim=0, keepdim=True
        )
        join_references.append(
            sequence_mean.expand(length, *sequence_mean.shape[1:])
        )
        start += length
    join_reference = torch.cat(join_references, dim=0)

    embedded, _ = model.gatr(
        multivectors,
        scalars=scalars,
        attention_mask=mask,
        join_reference=join_reference,
    )
    latent = embedded[:, 0, :]
    pieces = [model.clustering(latent), model.beta(latent)]
    if model.helix_proxy is not None:
        pieces.append(model.helix_proxy(latent))
    return torch.cat(pieces, dim=1)


def _run_event_loop(ev, model, dataset, device, max_events, batch_size,
                    max_batch_tokens):
    """Reference evaluator loop, additionally caching hit_type.

    Events are packed per forward as in training's TokenBudgetBatchSampler:
    up to ``batch_size`` events and (if max_batch_tokens > 0) up to
    ``max_batch_tokens`` hits, whichever comes first."""
    import dgl
    import torch

    events = []
    iterator = iter(dataset)
    pending_graphs = []
    pending_rows = []

    def flush():
        pending = [(g, r.detach().cpu().numpy())
                   for g, r in zip(pending_graphs, pending_rows)]
        pending_graphs.clear()
        pending_rows.clear()
        if not pending:
            return

        batched = len(pending) > 1 and batch_size > 1
        if batched:
            batch_graph = dgl.batch([g for g, _ in pending]).to(device)
            batched_output = _forward_batched(model, batch_graph, device)
        else:
            outputs = [model(g.to(device)) for g, _ in pending]
        n_forward_events = len(pending)
        n_forward_tokens = sum(int(g.num_nodes()) for g, _ in pending)

        offset = 0
        for index, (graph, rows_np) in enumerate(pending):
            if batched:
                n_nodes = int(graph.num_nodes())
                output = batched_output[offset : offset + n_nodes]
                offset += n_nodes
            else:
                output = outputs[index]
            expected = model.embedding_dim + 1
            if output.ndim != 2 or output.shape[1] < expected:
                raise SystemExit(
                    f"model output shape {tuple(output.shape)}; expected at "
                    f"least {expected} columns"
                )
            coords = output[:, : model.embedding_dim].float().cpu().numpy()
            beta = torch.sigmoid(
                output[:, model.embedding_dim]
            ).float().cpu().numpy()
            truth = (
                graph.ndata["particle_number"].detach().cpu().numpy().reshape(-1)
            )
            hit_type = (
                graph.ndata["hit_type"].detach().cpu().numpy().reshape(-1)
            )
            events.append(
                {
                    "coords": coords,
                    "beta": beta,
                    "truth": truth,
                    "hit_type": hit_type,
                    "particle_info": ev._particle_info(graph, rows_np),
                }
            )
        print(f"\r[inference] {len(events)} event(s) "
              f"(last forward: {n_forward_events} events, "
              f"{n_forward_tokens} hits)", end="", flush=True)

    pending_tokens = [0]

    with torch.inference_mode():
        while max_events <= 0 or len(events) + len(pending_graphs) < max_events:
            try:
                graph, particle_rows = next(iterator)
            except StopIteration:
                break
            # flush before this event if it would overflow the token budget
            n_nodes = int(graph.num_nodes())
            if (max_batch_tokens > 0 and pending_graphs
                    and pending_tokens[0] + n_nodes > max_batch_tokens):
                flush()
                pending_tokens[0] = 0
            pending_graphs.append(graph)
            pending_rows.append(particle_rows)
            pending_tokens[0] += n_nodes
            if len(pending_graphs) >= batch_size:
                flush()
                pending_tokens[0] = 0
        flush()
    print()
    if not events:
        raise SystemExit("dataset produced no events")
    return events


# ---------------------------------------------------------------------------
# clustering + matching -> per-particle cache
# ---------------------------------------------------------------------------
def events_to_particles(events, meta, args):
    """Per-particle rows: kinematics, SI/CDC hit split, matched flag."""
    repo = Path(args.training_repo).resolve()
    if str(repo.parent) not in sys.path:
        sys.path.insert(0, str(repo.parent))
    from shared_training import tracking_metrics as tm

    rows = []
    n_missing_info = 0
    for event_id, event in enumerate(events):
        labels = tm.greedy_cluster(
            event["beta"], event["coords"], args.tbeta, args.td, args.min_hits,
            rejected_seed_policy=args.rejected_seed_policy,
        )
        truth_ids, truth_counts, reco_ids, reco_counts, overlap = (
            tm._overlap_tables(labels, event["truth"], args.truth_min_hits)
        )
        matches = tm._one_to_one_matches(
            overlap, truth_counts, reco_counts, args.matching_metric
        )
        match_by_truth = {int(truth_ids[i]): int(j) for i, j in matches}
        truth_index = {int(t): i for i, t in enumerate(truth_ids)}

        truth = event["truth"]
        hit_type = event["hit_type"]
        all_ids = np.unique(truth[truth > 0])
        for tid in all_ids:
            tmask = truth == tid
            n_si = int((hit_type[tmask] == 1).sum())  # hit_type 1 = silicon
            n_cdc = int((hit_type[tmask] == 0).sum())  # hit_type 0 = drift
            info = event["particle_info"].get(int(tid))

            i = truth_index.get(int(tid))
            matched = i is not None and int(tid) in match_by_truth
            eff_hit = pur_match = 0.0
            if i is not None and overlap.shape[1]:
                j_best = int(np.argmax(overlap[i]))
                eff_hit = float(overlap[i, j_best] / max(truth_counts[i], 1))
                pur_match = float(overlap[i, j_best] / max(reco_counts[j_best], 1))
                if matched:
                    j = match_by_truth[int(tid)]
                    eff_hit = float(overlap[i, j] / max(truth_counts[i], 1))
                    pur_match = float(overlap[i, j] / max(reco_counts[j], 1))

            if info is None:
                n_missing_info += 1
            rows.append(
                {
                    "eventID": event_id,
                    "particleID": int(tid),
                    "pT": info["pt"] if info else np.nan,
                    "theta": info["theta"] if info else np.nan,  # rad
                    "genStatus": (int(round(info["gen_status"]))
                                  if info else np.nan),
                    "displacement": (info["displacement"] if info else np.nan),
                    "numSIhits": n_si,
                    "numCDChits": n_cdc,
                    "matched": bool(matched),
                    "hitEfficiency": eff_hit,
                    "hitPurity": pur_match,
                }
            )
    if n_missing_info:
        print(f"[matching] {n_missing_info} particles lacked kinematic info "
              "(left NaN, dropped by the cuts)")

    df = pd.DataFrame(rows)
    run = dict(meta)
    run.update(
        {
            "tbeta": args.tbeta,
            "td": args.td,
            "min_hits": args.min_hits,
            "rejected_seed_policy": args.rejected_seed_policy,
            "matching_metric": args.matching_metric,
            "truth_min_hits": args.truth_min_hits,
        }
    )
    return df, run


# ---------------------------------------------------------------------------
# cut application (plotting path, works on any per-particle cache)
# ---------------------------------------------------------------------------
def limit_error_bars(y, yerr, upper_limit=1.0):
    yerr_upper = np.minimum(y + yerr, upper_limit) - y
    return yerr, yerr_upper


def theta_to_float(x):
    """Tolerate theta stored as scalar or one-element list/ndarray."""
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return float(x[0]) if len(x) > 0 else None
    return float(x)


def _is_noise_label(x):
    return isinstance(x, (list, tuple, np.ndarray)) and 0 in x


def _label_matched(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return len(x) > 0
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    return int(x) > 0


def looks_like_particle_cache(columns) -> bool:
    cols = set(columns)
    return "matched" in cols or ("trackLabel" in cols and "pT" in cols)


def load_particle_cache(pattern, args):
    """Load per-particle parquet file(s) and normalise to the cache schema."""
    values = pattern if isinstance(pattern, list) else [pattern]
    files = [f for f in expand_input_files(values) if os.path.isfile(f)]
    if not files:
        raise SystemExit(f"no cache files match {pattern!r}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if len(files) > 1:
        print(f"[cache] concatenated {len(files)} cache files")
    if "matched" in df.columns:
        df["matched"] = df["matched"].astype(bool)
    elif "trackLabel" in df.columns:
        # reference convention: a match to the noise cluster 0 is unmatched
        noise = df["trackLabel"].apply(_is_noise_label)
        if noise.any():
            df.loc[noise, "trackLabel"] = pd.Series(
                [[] for _ in range(int(noise.sum()))],
                index=df.index[noise], dtype=object,
            )
        df["matched"] = df["trackLabel"].apply(_label_matched)
    if "theta" in df.columns:
        df["theta"] = df["theta"].apply(theta_to_float)
    return df


def apply_cuts(df, args):
    cut_cols = ["pT", "theta", "matched"]
    if args.gen_status is not None:
        cut_cols.append("genStatus")
    if args.min_si_hits is not None:
        cut_cols.append("numSIhits")
    if args.min_cdc_hits is not None:
        cut_cols.append("numCDChits")
    missing = [c for c in set(cut_cols) if c not in df.columns]
    if missing:
        raise SystemExit(
            f"requested cuts need columns {missing}; "
            f"available: {sorted(df.columns)}"
        )

    n_in = len(df)
    df = df.dropna(subset=list(dict.fromkeys(cut_cols)))
    print(f"[cut] dropna kinematics/match info: {n_in} -> {len(df)} rows")

    df["theta_deg"] = (
        df["theta"].astype(float)
        if args.theta_unit == "deg"
        else np.degrees(df["theta"].astype(float))
    )

    mask = (df["theta_deg"] > args.theta_min) & (df["theta_deg"] < args.theta_max)
    print(f"[cut] {args.theta_min:g} < theta < {args.theta_max:g} deg: "
          f"{int(mask.sum())} rows")
    mask &= df["pT"] > args.min_pt
    print(f"[cut] pT > {args.min_pt:g} GeV: {int(mask.sum())} rows")
    if args.gen_status is not None:
        m = df["genStatus"].isin(args.gen_status)
        print(f"[cut] genStatus in {args.gen_status}: "
              f"{int((mask & m).sum())} rows")
        mask &= m
    if args.min_si_hits is not None:
        m = df["numSIhits"] >= args.min_si_hits
        print(f"[cut] numSIhits >= {args.min_si_hits}: "
              f"{int((mask & m).sum())} rows")
        mask &= m
    if args.min_cdc_hits is not None:
        m = df["numCDChits"] >= args.min_cdc_hits
        print(f"[cut] numCDChits >= {args.min_cdc_hits}: "
              f"{int((mask & m).sum())} rows")
        mask &= m
    # shared-pipeline convention: only particles with >= truth_min_hits
    # detector hits are matchable; apply the same threshold to the denominator
    if {"numSIhits", "numCDChits"} <= set(df.columns):
        m = (df["numSIhits"] + df["numCDChits"]) >= args.truth_min_hits
        print(f"[cut] numSIhits + numCDChits >= {args.truth_min_hits}: "
              f"{int((mask & m).sum())} rows")
        mask &= m

    df = df[mask]
    if df.empty:
        raise SystemExit("no particles left after cuts - relax the selection")
    return df


def binned_efficiency(df, min_x, max_x, bin_step):
    """Binomial tracking efficiency in log-spaced pT bins."""
    bins = np.exp(np.arange(np.log(min_x), np.log(max_x), bin_step))
    if bins[-1] < max_x:
        bins = np.append(bins, max_x)
    bin_indices = np.digitize(df["pT"], bins)
    # close the range at max_x: particles exactly on the last edge belong to
    # the final bin (as in the reference evaluator)
    bin_indices = np.where(df["pT"].to_numpy() == bins[-1],
                           len(bins) - 1, bin_indices)

    centers, effs, errs, ns = [], [], [], []
    for i in range(1, len(bins)):
        df_bin = df[bin_indices == i]
        n_total = len(df_bin)
        centers.append((bins[i - 1] + bins[i]) / 2)
        if n_total == 0:
            effs.append(np.nan)
            errs.append(0.0)
            ns.append(0)
            continue
        n_matched = int(df_bin["matched"].sum())
        eff = n_matched / n_total
        effs.append(eff)
        errs.append(np.sqrt(eff * (1 - eff) / n_total))
        ns.append(n_total)

    return (bins, np.array(centers), np.array(effs),
            np.array(errs), np.array(ns))


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------
def make_plot(centers, effs, errs, args, run_info, n_events=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    ax.scatter(centers, effs, marker="s", c=[_TEAL] * len(centers),
               s=30, label=args.label, zorder=3)
    yerr_low, yerr_up = limit_error_bars(effs, errs, upper_limit=1)
    ax.errorbar(centers, effs, yerr=[yerr_low, yerr_up], ecolor=_TEAL,
                linestyle="none", capsize=4, zorder=2)

    ax.set_xlabel("$p_T$ [GeV]")
    ax.set_ylabel("Tracking efficiency")
    ax.set_xscale("log")
    ax.set_xlim([args.min_x, args.max_x])
    ax.set_ylim([0.01, 1.01])
    ax.legend(loc="lower right")

    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=10))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.minorticks_on()
    ax.grid(which="major", linestyle=":", linewidth=0.5, color="black")
    ax.grid(which="minor", linestyle=":", linewidth=0.5, color="gray")

    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.family"] = "serif"

    # selection textbox (reference-plot layout)
    lines = []
    sample_lines = (
        _SAMPLE_PRESETS[args.sample]
        if args.sample
        else (args.sample_text.split(r"\n") if args.sample_text else [])
    )
    lines.extend(sample_lines)
    if run_info is not None:
        metric = ("double majority" if run_info["matching_metric"]
                  == "double_majority" else run_info["matching_metric"])
        lines.append(
            rf"$t_\beta={run_info['tbeta']:g},~t_d={run_info['td']:g}$"
        )
        lines.append(rf"{metric}, {run_info['events']} events")
    elif n_events is not None:
        lines.append(f"{n_events} events")
    lines.append(rf"${args.theta_min:g}^\circ < \theta < {args.theta_max:g}^\circ$")
    extra_cuts = 1 if run_info is not None else 0
    if args.gen_status is not None:
        status = ", ".join(str(s) for s in args.gen_status)
        lines.append(rf"$genStatus \in [{status}]$")
        extra_cuts += 1
    if args.min_pt > 0:
        lines.append(rf"$p_T > {args.min_pt:g}~\mathrm{{GeV}}$")
        extra_cuts += 1
    if args.min_si_hits is not None:
        lines.append(rf"$N_\mathrm{{SI}} \geq {args.min_si_hits}$")
        extra_cuts += 1
    if args.min_cdc_hits is not None:
        lines.append(rf"$N_\mathrm{{CDC}} \geq {args.min_cdc_hits}$")
        extra_cuts += 1
    if args.min_si_hits is None and args.min_cdc_hits is None:
        lines.append(rf"$N_\mathrm{{hits}} \geq {args.truth_min_hits}$")

    # reference geometry fits ~5 short lines at fontsize 22; shrink and pull
    # left for the wide textboxes (run-info and/or sample + many cuts)
    if run_info is not None:
        x0, y0, fontsize = 0.30, 0.47, 15
    elif len(lines) <= 5:
        x0, y0 = (0.55, 0.5) if extra_cuts else (0.62, 0.33)
        fontsize = 22 if extra_cuts else 15
    else:
        x0, y0, fontsize = 0.35, 0.47, 15
    ax.text(
        x0, y0, "\n".join(lines),
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment="center",
        horizontalalignment="left",
        linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="none",
                  edgecolor="none"),
    )
    return fig


# standard process annotations for the plot textbox
_SAMPLE_PRESETS = {
    "zuds": [
        r"$Z/\gamma^* \rightarrow q\bar{q}\ (q=u,d,s)$",
        r"$m_Z = 91~\mathrm{GeV}$",
    ],
}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("parquet", nargs="+",
                   help="one or more parquet files or glob patterns (quote "
                        "globs or let the shell expand them; both work): "
                        "hit-level Graphs files, or per-particle cache "
                        "file(s) written by a previous inference run")
    p.add_argument("-o", "--output", default="tracking_efficiency_vs_pt.png",
                   help="output image path (default: %(default)s)")
    p.add_argument("--label", default="GGTF IDEA", help="legend label")

    # inference
    inf = p.add_argument_group("inference (hit-level Graphs input)")
    inf.add_argument("--checkpoint", default=None,
                     help="model checkpoint (.ckpt); required for Graphs input")
    inf.add_argument("--weights", choices=("ema", "raw", "auto"), default="ema",
                     help="checkpoint weights to use (default: %(default)s; "
                          "'auto' falls back to raw when no EMA is present)")
    inf.add_argument("--tbeta", type=float, default=0.7,
                     help="min sigmoid(beta) for a condensation-point seed "
                          "(default: %(default)s)")
    inf.add_argument("--td", type=float, default=0.1,
                     help="max embedding-space distance for hit assignment "
                          "(default: %(default)s)")
    inf.add_argument("--min-hits", type=int, default=3,
                     help="minimum reconstructed-track size in clustering "
                          "(default: %(default)s); inference only, has no "
                          "effect on a per-particle cache input")
    inf.add_argument("--rejected-seed-policy",
                     choices=("discard", "keep", "attach-after-accept"),
                     default="discard",
                     help="handling of seeds whose candidate is below --min-hits")
    inf.add_argument("--matching-metric",
                     choices=("double_majority", "idea", "hungarian"),
                     default="double_majority",
                     help="cluster/particle matching criterion (default: %(default)s)")
    inf.add_argument("--truth-min-hits", type=int, default=3,
                     help="min detector hits (numSIhits + numCDChits) for a "
                          "truth particle; match-eligibility at inference "
                          "time AND denominator cut at plotting time - this "
                          "is the textbox's N_hits >= K line (default: %(default)s)")
    inf.add_argument("--max-files", type=int, default=-1,
                     help="cap on input files (negative = all)")
    inf.add_argument("--max-events", type=int, default=-1,
                     help="global event cap (negative = all)")
    inf.add_argument("--device", default="cpu",
                     help="torch device (default: %(default)s; use e.g. "
                          "cuda or cuda:1 for GPU inference)")
    inf.add_argument("--num-shards", type=int, default=1,
                     help="split the input into N event shards (run N "
                          "processes, one per GPU, with --shard-id 0..N-1)")
    inf.add_argument("--shard-id", type=int, default=0,
                     help="this process's shard index [0, num-shards)")
    inf.add_argument("--batch-size", type=int, default=None,
                     help="events per forward pass. When set, this fully "
                          "controls packing and --max-batch-tokens is "
                          "IGNORED. Default: one event per forward, bitwise "
                          "the reference evaluator. Batches use a "
                          "block-diagonal attention mask (xformers on CUDA, "
                          "dense bool mask on CPU); on CPU the attention "
                          "score memory scales as hits^2, so large batches "
                          "need matching RAM")
    inf.add_argument("--max-batch-tokens", type=int, default=None,
                     help="hits packed per forward; only effective when "
                          "--batch-size is NOT set (then events per forward "
                          "are unlimited, token-driven packing like "
                          "training's TokenBudgetBatchSampler)")
    inf.add_argument("--num-threads", type=int, default=0,
                     help="torch CPU threads (default: 0 = torch default, "
                          "all cores)")
    inf.add_argument("--data-config", default=None,
                     help="data YAML (default: picked from the checkpoint's "
                          "geometry-only or detector-feature architecture)")
    inf.add_argument("--layers-per-superlayer", type=int, nargs=14, default=None,
                     metavar="N", help="override the checkpoint's 14-entry layout")
    inf.add_argument("--position-scale", type=float, default=None,
                     help="override checkpoint metadata (default 1000)")
    inf.add_argument("--training-repo", default=str(_DEFAULT_REPO),
                     help="path to model_training/GATR_CIRCE_LOSS "
                          "(default: ../model_training/GATR_CIRCE_LOSS "
                          "relative to this script)")
    inf.add_argument("--save-cache", default=None,
                     help="where to write the per-particle cache parquet "
                          "(default: <output stem>_particles.parquet)")

    # particle cuts
    cuts = p.add_argument_group("particle cuts")
    cuts.add_argument("--gen-status", type=int, nargs="+", default=None,
                      metavar="S", help="keep particles with genStatus in this list")
    cuts.add_argument("--min-pt", type=float, default=0.0,
                      help="keep particles with pT > MIN_PT [GeV] (default: %(default)s)")
    cuts.add_argument("--min-si-hits", type=int, default=None, metavar="N",
                      help="keep particles with numSIhits >= N "
                           "(hit_type==1 count)")
    cuts.add_argument("--min-cdc-hits", type=int, default=None, metavar="N",
                      help="keep particles with numCDChits >= N "
                           "(hit_type==0 count)")
    cuts.add_argument("--theta-min", type=float, default=10.0,
                      help="theta > THETA_MIN [deg] (default: %(default)s)")
    cuts.add_argument("--theta-max", type=float, default=170.0,
                      help="theta < THETA_MAX [deg] (default: %(default)s)")
    cuts.add_argument("--theta-unit", choices=("rad", "deg"), default="rad",
                      help="unit of the theta column (default: %(default)s)")

    # binning / cosmetics
    p.add_argument("--min-x", type=float, default=0.1,
                   help="lower pT bin edge [GeV] (default: %(default)s)")
    p.add_argument("--max-x", type=float, default=60.0,
                   help="upper pT bin edge [GeV] (default: %(default)s)")
    p.add_argument("--bin-step", type=float, default=0.2,
                   help="bin width in natural-log pT (default: %(default)s)")
    p.add_argument("--sample", choices=tuple(_SAMPLE_PRESETS), default=None,
                   help="process annotation shown in the textbox; 'zuds' "
                        "prints the standard Z/gamma* -> qq (q=u,d,s), "
                        "m_Z = 91 GeV lines")
    p.add_argument("--sample-text", default=None,
                   help=r"custom textbox lines, separated by literal \n "
                        r"(overrides --sample if both are given)")
    parsed = p.parse_args(argv)
    parsed._parser = p  # for defaults lookup (cache-mode warnings)
    return parsed


def main(argv=None):
    args = parse_args(argv)

    if args.min_x <= 0 or args.max_x <= args.min_x or args.bin_step <= 0:
        raise SystemExit("require 0 < --min-x < --max-x and --bin-step > 0")
    if args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("require --batch-size >= 1")
    if args.max_batch_tokens is not None and args.max_batch_tokens < 1:
        raise SystemExit("require --max-batch-tokens >= 1")
    if args.num_shards < 1 or not (0 <= args.shard_id < args.num_shards):
        raise SystemExit("require --num-shards >= 1 and "
                         "0 <= --shard-id < --num-shards")

    out_path = Path(args.output)

    values = args.parquet if isinstance(args.parquet, list) else [args.parquet]
    matches = [f for f in expand_input_files(values) if os.path.isfile(f)]
    if not matches:
        raise SystemExit(f"no file matches {args.parquet!r}")

    import pyarrow.parquet as pq

    columns = [c.name for c in pq.ParquetFile(matches[0]).schema]
    run_info = None

    if looks_like_particle_cache(columns):
        print(f"[mode] per-particle cache detected: {matches[0]}")
        defaults = {a.dest: a.default
                    for a in args._parser._actions if a.dest != "help"}
        ignored = [f"--{d.replace('_', '-')}" for d in _INFERENCE_ONLY_OPTS
                   if getattr(args, d, None) != defaults.get(d)]
        if ignored:
            print("[mode] warning: inference option(s) have no effect on a "
                  "cache (matching was already fixed at inference time): "
                  + ", ".join(ignored))
        df = load_particle_cache(args.parquet, args)
    else:
        print(f"[mode] hit-level Graphs input detected: {matches[0]}")
        if args.checkpoint is None:
            raise SystemExit(
                "hit-level Graphs input requires --checkpoint to run inference"
            )
        events, meta = run_inference(args)
        df, run_info = events_to_particles(events, meta, args)

        cache_path = args.save_cache or str(
            out_path.with_name(out_path.stem + "_particles.parquet")
        )
        df.to_parquet(cache_path)
        with open(Path(cache_path).with_suffix(".json"), "w") as fh:
            json.dump(run_info, fh, indent=2, sort_keys=True)
        print(f"[cache] wrote {cache_path} ({len(df)} particles) + .json")

    df = apply_cuts(df, args)
    bins, centers, effs, errs, ns = binned_efficiency(
        df, args.min_x, args.max_x, args.bin_step
    )

    n_events = (int(df["eventID"].nunique())
                if "eventID" in df.columns else None)

    total = int(ns.sum())
    matched = int(round(float(np.nansum(effs * ns))))
    if total:
        print(f"\nselection: {total} particles in "
              f"[{args.min_x:g}, {args.max_x:g}] GeV"
              + (f" from {n_events} events" if n_events is not None else "")
              + f", {matched} matched, "
              f"overall eff = {matched / total:.4f}")
    print(f"{'pT bin [GeV]':>22} {'N':>7} {'eff':>9}")
    for lo, hi, e, er, n in zip(bins[:-1], bins[1:], effs, errs, ns):
        if n == 0:
            continue
        print(f"[{lo:9.3f}, {hi:9.3f}] {n:7d} {e:7.4f} +/- {er:.4f}")

    fig = make_plot(centers, effs, errs, args, run_info, n_events)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    sys.exit(main())
