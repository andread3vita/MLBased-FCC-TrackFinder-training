"""C-GATr FCC LightningModule (`cgatr_fcc`).

The model definition stays local to CIRCE. The loss, warmup helpers, tracking
metrics, plots, and logging contracts are imported from ``shared_training``.

  * M1-M5 are baked into src.model; no env flags needed.
  * Same EMA(0.999) over the full state_dict (BatchNorm buffers included).
  * Same AdamW(weight_decay=1e-4) + LambdaLR with linear warmup over
    warmup_epochs then half-cosine decay to min_lr.
  * Same OC loss with beta_suppress=0.1, qmin=0.1, var_weight=0.3
    ramped linearly over var_warmup_epochs (epoch-based, 1-indexed).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import lightning as L
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from src.model import CGATrParquetModel
from shared_training.circe_loss import (
    object_condensation_loss, sequence_lengths_to_batch, variance_weight,
)
from shared_training.logging_contract import (
    log_sweep_metrics, log_training_metrics, log_validation_loss,
)
from shared_training.tracking_metrics import (
    TRACKING_COUNT_KEYS, TRACKING_DISPLACEMENT_BINS, TRACKING_PT_BINS,
    matching_comparison_binned_counts, matching_comparison_plot_series,
    parse_grid, run_operating_point_sweep,
    save_sweep_results, save_tracking_efficiency_displacement_plot,
    save_tracking_efficiency_pt_plot, tracking_metrics_from_counts,
)
from shared_training.wandb_logger import (
    log_wandb_media, operating_point_media, tracking_efficiency_media,
    wandb_html,
)
from shared_training.validation_media import validation_event_media


def relabel_small_targets(mc_index_loss, batch_ids, noise_index, min_hits):
    """GGTF's `create_garbage_label(..., minNumHits)`, expressed as relabelling.

    A particle with fewer than `min_hits` hits *in its own event* stops being a target and
    becomes noise. Its hits stay in the input: deleting them would be the looper filter,
    which is a different and non-deployable thing (M23).

    Theirs runs during graph construction and drops the particles from `y_data_graph`
    (`functions_graph_tracking.py:156`), so their loss never sees a sub-3-hit target. Ours
    had the rule only in the scorer, so we trained against 146.6 primaries an event where
    they train against 33.2 — roughly 113 one-hit stubs an event that their model learns to
    suppress and ours learned to reconstruct. See M33.

    `min_hits` of 0 or 1 is a no-op, which is the behaviour of every run before 2026-08-08.
    """
    if min_hits <= 1:
        return mc_index_loss

    signal = mc_index_loss != noise_index
    if not bool(signal.any()):
        return mc_index_loss

    # Keyed on (event, particle): the same index in two events is two different particles.
    pair = torch.stack([batch_ids[signal], mc_index_loss[signal]], dim=1)
    _, inverse, counts = torch.unique(pair, dim=0, return_inverse=True, return_counts=True)
    undersized = counts[inverse] < min_hits
    mc_index_loss[torch.nonzero(signal, as_tuple=True)[0][undersized]] = noise_index
    return mc_index_loss


class EMAShadow:
    """EMA over the full state_dict (parameters + BN running stats).

    Floating-point tensors are decayed; integer buffers (e.g.
    `num_batches_tracked`) are copied. Stored as a flat state_dict so
    it round-trips through Lightning checkpoints transparently under
    the key `ema_state_dict`.
    """

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        sd = model.state_dict()
        for k, v in sd.items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(
                    v.detach(), alpha=1.0 - self.decay,
                )
            else:
                self.shadow[k].copy_(v.detach())

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.detach().clone() for k, v in state_dict.items()}


class CGATrV35LightningModule(L.LightningModule):
    """Lightning wrapper around CGATrParquetModel (M1-M5 baked in) + OC loss."""

    def __init__(self, args, steps_per_epoch: Optional[int] = None):
        super().__init__()
        # Persist serializable hparams so `Trainer.fit(..., ckpt_path="last")`
        # can resume without re-running the CLI.
        self.save_hyperparameters(
            {k: v for k, v in vars(args).items()
             if isinstance(v, (int, float, str, bool, type(None)))}
        )
        self.args = args
        self.model = CGATrParquetModel(args)

        # `steps_per_epoch` drives the LambdaLR warmup + cosine schedule.
        # Default (None / 0): resolved in `configure_optimizers` from
        # `trainer.estimated_stepping_batches` — that's the safe path,
        # because it accounts for DDP sharding, `limit_train_batches`,
        # and grad accumulation.
        self._steps_per_epoch = int(steps_per_epoch) if steps_per_epoch else 0

        self._ema: Optional[EMAShadow] = None
        self._ema_decay = float(getattr(args, "ema_decay", 0.0) or 0.0)
        self._saved_train_state: Optional[Dict[str, torch.Tensor]] = None
        self._pending_ema_state: Optional[Dict[str, torch.Tensor]] = None

        self._val_loss_sum: float = 0.0
        self._val_loss_event_count: int = 0
        self._val_metrics: List[Dict[str, float]] = []
        self._val_tracking_events = []
        self._validation_working_points = None
        self._saving_validation_sweep_checkpoint = False
        self._train_loss_sum: float = 0.0
        self._train_loss_n: int = 0
        self._train_loss_sum_t: Optional[torch.Tensor] = None
        self._last_train_batch_finite = False
        self._finite_batches_since_optimizer_step = 0

    @property
    def embedding_dim(self) -> int:
        """Alias for downstream evaluation scripts expecting `model.embedding_dim`."""
        return self.args.embed_dim

    def split_output(self, output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split model output into condensation coordinates and beta logits."""
        return self.model.split_output(output)

    def forward(self, features, seq_lens):
        return self.model(features, seq_lens)

    # ---- fit lifecycle ------------------------------------------------------
    def on_fit_start(self):
        if self._ema is None and self._ema_decay > 0.0:
            self._ema = EMAShadow(self.model, decay=self._ema_decay)
            if self._pending_ema_state is not None:
                tgt = next(self.model.parameters()).device
                self._ema.load_state_dict({
                    k: v.to(tgt) for k, v in self._pending_ema_state.items()
                })
                self._pending_ema_state = None

        if self.trainer is not None and self.trainer.is_global_zero:
            print(
                f"[cgatr_fcc] start  | num_epochs={self.args.num_epochs}, "
                f"steps/epoch={self._steps_per_epoch}, "
                f"warmup_epochs={self.args.warmup_epochs}, "
                f"ema_decay={self._ema_decay}, "
                f"world_size={self.trainer.world_size}",
                flush=True,
            )

    # ---- shared step --------------------------------------------------------
    def _shared_step(self, batch) -> Dict[str, torch.Tensor]:
        features = batch["features"]
        mc_index = batch["mc_index"]
        is_secondary = batch["is_secondary"]
        seq_lens = batch["seq_lens"]
        batch_ids = sequence_lengths_to_batch(seq_lens, features.device)

        output = self.model(features, seq_lens)

        ed = self.args.embed_dim
        mc_index_loss = mc_index.clone()
        # A no-op on this dataset, kept because the column is part of the schema.
        # produced_by_secondary is zero for all 16M hits checked across eight
        # seeds and both subdetectors, so there is no secondary population to
        # relabel and no --keep_secondaries flag is needed to match GGTF, whose
        # active path keeps secondaries as targets.
        #
        # `noise_index` is where `--fix_particle_zero` acts. The dataset has no
        # unassociated hits whatsoever -- every one of 2.0M checked maps to a real
        # particle in its own event -- so treating index 0 as noise does not label
        # noise, it labels *particle 0*, which is a generator-status-1 particle
        # present in every event and charged in about a fifth of them. Those are
        # full tracks, a median 122 hits at a median 0.84 GeV, and the default
        # trains against them: excluded from the attractive term and from signal
        # beta, repelled from every object, and beta actively suppressed by
        # L_beta_noise. Moving the sentinel to -1, a value no hit carries, makes
        # them targets and empties the noise class, which is what the data says it
        # should be. Off by default only so the in-flight ladder stays internally
        # comparable; see FINDINGS.md M20.
        noise_index = -1 if getattr(self.args, "fix_particle_zero", False) else 0
        mc_index_loss[is_secondary] = noise_index

        # GGTF's `create_garbage_label(..., minNumHits=3)`, on our side of the fence.
        #
        # Theirs runs during graph construction (`functions_graph_tracking.py:156`) and then
        # drops those particles from `y_data_graph`, so their loss never sees a target with
        # fewer than three hits, while the hits themselves stay in the graph as noise. We had
        # the rule only in the scorer, which meant we trained against 146.6 primaries an event
        # where they train against 33.2 -- about 113 one-hit stubs an event that their model
        # learns to suppress and ours learned to reconstruct. That is the likeliest source of
        # the fragmentation in M27, since a model rewarded for one-hit clusters emits them and
        # each becomes a fake or a clone under their counting. See M33.
        #
        # Relabelling, never deleting: dropping the hits would be the looper filter, which is a
        # different and non-deployable thing (M23). The hits stay in the input, they simply
        # stop being objects the loss has to condense.
        mc_index_loss = relabel_small_targets(
            mc_index_loss, batch_ids, noise_index,
            int(getattr(self.args, "min_target_hits", 0) or 0),
        )

        coords = output[:, :ed].float()
        if self.args.cosine_norm:
            coords = F.normalize(coords, dim=-1)
        beta_val = torch.sigmoid(output[:, ed].float())

        return {
            "coords": coords,
            "beta_val": beta_val,
            "mc_index": mc_index,
            "mc_index_loss": mc_index_loss,
            "is_secondary": is_secondary,
            "seq_lens": seq_lens,
            "batch_ids": batch_ids,
            "output": output,
            "positions": batch.get("positions"),
            "noise_index": noise_index,
            "track_separation_weight": batch.get("track_separation_weight"),
            "particle_info": batch.get("particle_info"),
        }

    # ---- training step ------------------------------------------------------
    def _dummy_ddp_step(self) -> torch.Tensor:
        """Zero-loss forward on a 2-hit dummy event. Keeps DDP allreduce balanced.

        The width has to follow the model's own flags rather than being fixed at 10.
        `--use_time` adds a column and GGTF's projective encoding adds three more for the
        drift direction, and that encoding raises outright when they are absent -- so a
        fixed-width dummy turns an empty batch, which this method exists to survive, into
        a crash on exactly the arms that need it most. The empty batches come from
        `--drop_loopers` filtering an event below four hits, which is Phase 1, whose
        projective arm carries that encoding.
        """
        params = next(self.model.parameters())
        n_cols = (10
                  + (1 if getattr(self.model, "use_time", False) else 0)
                  + (3 if getattr(self.model, "needs_drift_dir", False) else 0))
        dummy = torch.zeros(2, n_cols, device=params.device, dtype=params.dtype)
        dummy[:, 3] = 1.0
        if getattr(self.model, "needs_drift_dir", False):
            # A unit drift direction. Zeros would survive the 1e-8 guard in the encoding
            # but leave the hit at the wire, which is a degenerate geometry to hand a
            # backbone even for a discarded step.
            dummy[:, -1] = 1.0
        out = self.model(dummy, [2])
        return out.sum() * 0.0

    def on_train_epoch_start(self):
        self._train_loss_sum = 0.0
        self._train_loss_n = 0
        self._train_loss_sum_t = None

    def _zero_loss_for_skipped_batch(self) -> torch.Tensor:
        """Return a finite zero connected to every trainable DDP parameter."""
        parameter = next(self.parameters())
        zero_loss = parameter.new_zeros(())
        for parameter in self.parameters():
            if parameter.requires_grad and parameter.numel() > 0:
                zero_loss = zero_loss + parameter.reshape(-1)[0] * 0.0
        return zero_loss

    def training_step(self, batch, batch_idx):
        self._last_train_batch_finite = False
        if batch is None:
            return self._dummy_ddp_step()

        s = self._shared_step(batch)
        n_events = len(s["seq_lens"])
        vw = variance_weight(
            self.current_epoch + 1,
            self.args.var_weight,
            self.args.var_warmup_epochs,
        )

        loss, comp = object_condensation_loss(
            coords=s["coords"],
            beta=s["beta_val"],
            mc_index=s["mc_index_loss"].long(),
            batch=s["batch_ids"].long(),
            noise_index=s["noise_index"],
            qmin=self.args.qmin,
            attr_weight=self.args.attr_weight,
            repul_weight=self.args.repul_weight,
            fill_loss_weight=self.args.fill_loss_weight,
            use_average_cc_pos=self.args.use_average_cc_pos,
            beta_suppress_weight=self.args.beta_suppress_weight,
            var_weight=vw,
            return_components=True,
            oc_mode=self.args.oc_mode,
            track_separation_weight=s["track_separation_weight"],
        )

        local_bad = (~torch.isfinite(loss)) | (~torch.isfinite(s["output"]).all())
        global_bad = local_bad.to(dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_bad, op=dist.ReduceOp.MAX)
        if bool(global_bad.item()):
            parameters_finite = torch.stack([
                torch.isfinite(parameter.detach()).all()
                for parameter in self.parameters()
                if parameter.requires_grad
            ]).all().to(dtype=torch.int32, device=loss.device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(parameters_finite, op=dist.ReduceOp.MIN)
            if not bool(parameters_finite.item()):
                raise FloatingPointError(
                    "Model parameters became non-finite; refusing to skip a "
                    "batch after model state corruption"
                )
            self.log("train/nan_skip", 1.0, on_step=True, on_epoch=False,
                     prog_bar=False, sync_dist=False, logger=False,
                     batch_size=n_events)
            return self._zero_loss_for_skipped_batch()

        loss_d = loss.detach()
        if self._train_loss_sum_t is None:
            self._train_loss_sum_t = loss_d.double().clone()
        else:
            self._train_loss_sum_t += loss_d.double()
        self._train_loss_n += 1

        log_training_metrics(
            self,
            loss_d,
            comp,
            self.args.attr_weight,
            self.args.repul_weight,
            vw,
            n_events,
        )
        self._last_train_batch_finite = True
        self._finite_batches_since_optimizer_step += 1
        return loss

    # gradient clipping is handled by Trainer(gradient_clip_val=1.0)

    def on_before_optimizer_step(self, optimizer):
        if self._finite_batches_since_optimizer_step == 0:
            optimizer.zero_grad(set_to_none=True)

    # ---- validation lifecycle ----------------------------------------------
    def on_validation_epoch_start(self):
        self._validation_working_points = None
        if self._ema is not None:
            self._saved_train_state = {
                k: v.detach().clone() for k, v in self.model.state_dict().items()
            }
            self.model.load_state_dict(self._ema.state_dict())
        self._val_loss_sum = 0.0
        self._val_loss_event_count = 0
        self._val_metrics = []
        self._val_tracking_events = []

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return
        s = self._shared_step(batch)

        loss = object_condensation_loss(
            coords=s["coords"],
            beta=s["beta_val"],
            mc_index=s["mc_index_loss"].long(),
            batch=s["batch_ids"].long(),
            noise_index=s["noise_index"],
            qmin=self.args.qmin,
            attr_weight=self.args.attr_weight,
            repul_weight=self.args.repul_weight,
            beta_suppress_weight=self.args.beta_suppress_weight,
            var_weight=float(self.args.var_weight),
            oc_mode=self.args.oc_mode,
            track_separation_weight=s["track_separation_weight"],
        )
        if torch.isfinite(loss):
            n_events = len(s["seq_lens"])
            self._val_loss_sum += float(loss.item()) * n_events
            self._val_loss_event_count += n_events

        # Cache exactly the representation accepted by GATr's current metric
        # code. Truth IDs are positive and event-local; zero denotes noise.
        coords = s["coords"].detach().float().cpu().numpy()
        beta = s["beta_val"].detach().float().cpu().numpy()
        truth = s["mc_index_loss"].detach().cpu().numpy()
        raw_truth = s["mc_index"].detach().cpu().numpy()
        positions = (
            s["positions"].detach().float().cpu().numpy()
            if s["positions"] is not None else None
        )
        offset = 0
        cached_particle_info = s["particle_info"] or [
            {} for _ in s["seq_lens"]
        ]
        for event_slot, length in enumerate(s["seq_lens"]):
            end = offset + int(length)
            raw = truth[offset:end]
            event_truth = np.zeros(raw.shape, dtype=np.int64)
            signal = raw != s["noise_index"]
            event_particle_info = {}
            if np.any(signal):
                raw_ids, inverse = np.unique(raw[signal], return_inverse=True)
                event_truth[signal] = inverse + 1
                source_info = cached_particle_info[event_slot]
                event_particle_info = {
                    local_id: source_info[int(raw_id)]
                    for local_id, raw_id in enumerate(raw_ids, start=1)
                    if int(raw_id) in source_info
                }
            self._val_tracking_events.append({
                "positions": (
                    positions[offset:end] if positions is not None else None
                ),
                "coords": coords[offset:end],
                "beta": beta[offset:end],
                "truth": event_truth,
                "mc_particle_id": raw_truth[offset:end],
                "particle_info": event_particle_info,
            })
            offset = end

    def on_validation_epoch_end(self):
        device = next(self.model.parameters()).device
        loss_stats = torch.tensor(
            [self._val_loss_sum, self._val_loss_event_count],
            dtype=torch.float64,
            device=device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
        global_loss_sum, global_loss_event_count = loss_stats.tolist()
        avg_loss = global_loss_sum / max(global_loss_event_count, 1.0)
        tbetas = parse_grid(self.args.sweep_tbeta_grid, float)
        tds = parse_grid(self.args.sweep_td_grid, float)
        min_hits = parse_grid(self.args.sweep_min_hits_grid, int)
        max_events = int(self.args.validation_sweep_max_events)
        world_size = max(int(self.trainer.world_size), 1)
        rank = int(self.trainer.global_rank)
        base, remainder = divmod(max_events, world_size)
        local_events = self._val_tracking_events[:base + int(rank < remainder)]
        local_rows = run_operating_point_sweep(
            local_events, tbetas, tds, min_hits,
            metric=self.args.sweep_match_metric,
            truth_min_hits=int(self.args.sweep_truth_min_hits),
            rejected_seed_policy=self.args.rejected_seed_policy,
        )
        count_tensor = torch.tensor(
            [[row[key] for key in TRACKING_COUNT_KEYS] for row in local_rows],
            dtype=torch.int64, device=device,
        )
        event_count = torch.tensor(len(local_events), dtype=torch.int64, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(event_count, op=dist.ReduceOp.SUM)
        output_dir = Path(self.args.output_dir) / "validation_sweeps" / (
            f"epoch_{self.current_epoch:04d}"
        )
        working_points = None
        if self.trainer.is_global_zero and int(event_count.item()) > 0:
            rows = []
            for template, counts in zip(local_rows, count_tensor.cpu().tolist()):
                rows.append(tracking_metrics_from_counts(
                    template["tbeta"], template["td"], template["min_hits"],
                    dict(zip(TRACKING_COUNT_KEYS, counts)),
                ))
            working_points = save_sweep_results(
                rows, str(output_dir),
                metadata={
                    "implementation": "current_GATR_tracking_metrics",
                    "events": int(event_count.item()),
                    "matching": self.args.sweep_match_metric,
                    "rejected_seed_policy": self.args.rejected_seed_policy,
                },
                fixed_min_hits=int(self.args.sweep_truth_min_hits),
            )
        if dist.is_available() and dist.is_initialized():
            holder = [working_points]
            dist.broadcast_object_list(holder, src=0)
            working_points = holder[0]
        if working_points:
            self._validation_working_points = {
                name: dict(working_point)
                for name, working_point in working_points.items()
            }
            pareto = working_points["pareto_f1"]
            avg_eff = float(pareto["efficiency"])
            avg_fake = float(pareto["fake_rate"])
        else:
            avg_eff = avg_fake = 0.0

        log_validation_loss(self, avg_loss, batch_size=1)
        self.log(
            "val_loss", avg_loss, on_epoch=True, sync_dist=True,
            logger=False, batch_size=1,
        )
        if working_points:
            log_sweep_metrics(self, working_points)
            # Logger-hidden aliases used only by the common validation
            # checkpoint filename.
            self.log(
                "val_pareto_f1",
                float(working_points["pareto_f1"]["f1"]),
                on_epoch=True,
                sync_dist=True,
                logger=False,
                batch_size=1,
            )
            self.log(
                "val_max_tracking_efficiency",
                float(working_points["max_efficiency"]["efficiency"]),
                on_epoch=True,
                sync_dist=True,
                logger=False,
                batch_size=1,
            )
            comparison_order, comparison_rows, missing = (
                matching_comparison_binned_counts(
                    self._val_tracking_events,
                    working_points,
                    {
                        "pt": TRACKING_PT_BINS,
                        "displacement": TRACKING_DISPLACEMENT_BINS,
                    },
                    truth_min_hits=int(self.args.sweep_truth_min_hits),
                    min_theta=10.0,
                    max_theta=170.0,
                    gen_status=(0, 1),
                    rejected_seed_policy=self.args.rejected_seed_policy,
                )
            )
            pt_count_tensor = torch.tensor(
                np.stack([
                    np.stack(pair, axis=0)
                    for pair in comparison_rows["pt"]
                ], axis=0),
                dtype=torch.int64,
                device=device,
            )
            displacement_count_tensor = torch.tensor(
                np.stack([
                    np.stack(pair, axis=0)
                    for pair in comparison_rows["displacement"]
                ], axis=0),
                dtype=torch.int64,
                device=device,
            )
            missing_tensor = torch.tensor(
                [missing["pt"], missing["displacement"]],
                dtype=torch.int64,
                device=device,
            )
            binned_event_count = torch.tensor(
                len(self._val_tracking_events), dtype=torch.int64, device=device
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(pt_count_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(displacement_count_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(missing_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(binned_event_count, op=dist.ReduceOp.SUM)

            pt_image_paths = {}
            displacement_image_paths = {}
            if self.trainer.is_global_zero:
                pt_counts = pt_count_tensor.cpu().numpy()
                displacement_counts = displacement_count_tensor.cpu().numpy()
                for point_name in ("max_efficiency", "pareto_f1"):
                    pt_image_paths[point_name] = save_tracking_efficiency_pt_plot(
                        matching_comparison_plot_series(
                            comparison_order, pt_counts, point_name
                        ),
                        str(output_dir),
                        filename_stem=(
                            f"tracking_efficiency_vs_pt_{point_name}"
                        ),
                        bins=TRACKING_PT_BINS,
                        min_x=0.1,
                        max_x=60.0,
                        min_theta=10.0,
                        max_theta=170.0,
                        gen_status=(0, 1),
                    )
                    displacement_image_paths[point_name] = (
                        save_tracking_efficiency_displacement_plot(
                            matching_comparison_plot_series(
                                comparison_order,
                                displacement_counts,
                                point_name,
                            ),
                            str(output_dir),
                            filename_stem=(
                                "tracking_efficiency_vs_displacement_"
                                f"{point_name}"
                            ),
                            bins=TRACKING_DISPLACEMENT_BINS,
                            min_x=0.0,
                            max_x=2000.0,
                            min_theta=10.0,
                            max_theta=170.0,
                            gen_status=(0, 1),
                        )
                    )
                print(
                    "Saved double-majority and Hungarian tracking efficiency "
                    "versus pT and uniformly binned displacement from "
                    f"{int(binned_event_count.item())} validation events; "
                    "missing pT/displacement metadata for "
                    f"{int(missing_tensor[0].item())}/"
                    f"{int(missing_tensor[1].item())} eligible truth tracks.",
                    flush=True,
                )
            if self.trainer.is_global_zero and hasattr(self.logger, "log_image"):
                media = operating_point_media(output_dir)
                media.update(tracking_efficiency_media(
                    pt_image_paths, displacement_image_paths
                ))
                if (
                    self._val_tracking_events
                    and self._val_tracking_events[0]["positions"] is not None
                ):
                    hit_html = validation_event_media(
                        self._val_tracking_events[0],
                        working_points["pareto_f1"],
                        rejected_seed_policy=self.args.rejected_seed_policy,
                        output_dir=output_dir,
                        include_embedding=False,
                    )
                    media.update({
                        key: wandb_html(html) for key, html in hit_html.items()
                    })
                log_wandb_media(
                    self.logger,
                    media,
                )

        if self.trainer.is_global_zero:
            tag = ("[sanity]" if self.trainer.sanity_checking
                   else f"Epoch {self.current_epoch + 1}")
            print(
                f"  {tag} | Val Loss: {avg_loss:.4f} | "
                f"GATr metric: efficiency={avg_eff:.3f}, fake={avg_fake:.3f} | "
                f"({int(global_loss_event_count)} events)",
                flush=True,
            )

        if self._saved_train_state is not None:
            self.model.load_state_dict(self._saved_train_state)
            self._saved_train_state = None

        if not self.trainer.sanity_checking:
            device = next(self.model.parameters()).device
            if self._train_loss_sum_t is not None:
                loss_sum = self._train_loss_sum_t.to(device=device,
                                                     dtype=torch.float64)
            else:
                loss_sum = torch.tensor(self._train_loss_sum, device=device,
                                        dtype=torch.float64)
            loss_n = torch.tensor(float(self._train_loss_n), device=device,
                                  dtype=torch.float64)
            if (self.trainer.world_size > 1
                    and torch.distributed.is_available()
                    and torch.distributed.is_initialized()):
                torch.distributed.all_reduce(loss_sum)
                torch.distributed.all_reduce(loss_n)
            mean = (loss_sum / loss_n.clamp(min=1.0)).item()
            self.log("train_loss", mean, on_epoch=True, sync_dist=False,
                     logger=False, batch_size=1)
            self.trainer.callback_metrics["train_loss"] = torch.as_tensor(mean)
            self._train_loss_sum = 0.0
            self._train_loss_n = 0
            self._train_loss_sum_t = None

    # ---- EMA + ckpt hooks ---------------------------------------------------
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self._ema is not None and self._last_train_batch_finite:
            self._ema.update(self.model)

    def on_save_checkpoint(self, checkpoint):
        if self._ema is not None:
            checkpoint["ema_state_dict"] = self._ema.state_dict()
        if self._saving_validation_sweep_checkpoint:
            if not self._validation_working_points:
                raise RuntimeError(
                    "Cannot save a validation sweep checkpoint before "
                    "the operating-point sweep has completed"
                )
            checkpoint["validation_working_points"] = {
                name: dict(working_point)
                for name, working_point in self._validation_working_points.items()
            }

    def on_load_checkpoint(self, checkpoint):
        if "ema_state_dict" in checkpoint:
            if self._ema is not None:
                self._ema.load_state_dict(checkpoint["ema_state_dict"])
            else:
                self._pending_ema_state = checkpoint["ema_state_dict"]

    # ---- optimizer + scheduler (matches v35) -------------------------------
    def _resolve_steps_per_epoch(self) -> int:
        if self._steps_per_epoch and self._steps_per_epoch > 0:
            return self._steps_per_epoch
        if self.trainer is None:
            raise RuntimeError(
                "configure_optimizers called without a trainer; pass "
                "`steps_per_epoch` to CGATrV35LightningModule.__init__ "
                "instead (e.g. in a unit test)."
            )
        total = self.trainer.estimated_stepping_batches
        per_epoch = int(total // max(self.args.num_epochs, 1))
        if per_epoch <= 0:
            raise RuntimeError(
                f"trainer.estimated_stepping_batches={total} is too "
                f"small for num_epochs={self.args.num_epochs}"
            )
        return per_epoch

    def configure_optimizers(self):
        steps_per_epoch = self._resolve_steps_per_epoch()
        self._steps_per_epoch = steps_per_epoch

        _prec = str(getattr(self.args, "precision", "32-true"))
        # Lightning cannot clip gradients for a fused optimizer under AMP
        # because fused Adam/AdamW performs gradient unscaling internally.
        # Keep the faster fused path for true FP32, and use the standard
        # optimizer implementation for 16-mixed and bf16-mixed.
        _use_fused = _prec == "32-true"
        # Adam rather than AdamW exists for the GGTF parity runs: their training
        # uses plain Adam, and decoupled weight decay is not a neutral
        # difference when the comparison is meant to be to their recipe.
        _opt = str(getattr(self.args, "optimizer", "adamw")).lower()
        if _opt == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=float(self.args.start_lr), fused=_use_fused)
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=float(self.args.start_lr),
                weight_decay=float(getattr(self.args, "weight_decay", 1e-4)),
                fused=_use_fused,
            )

        total_steps = self.args.num_epochs * steps_per_epoch
        warmup_steps = (
            self.args.warmup_steps
            if getattr(self.args, "warmup_steps", None) is not None
            else self.args.warmup_epochs * steps_per_epoch
        )

        # Stored for the manual warmup in optimizer_step (plateau schedule).
        self._lr_schedule = str(getattr(self.args, "lr_schedule", "cosine"))
        self._base_lr = float(self.args.start_lr)
        self._min_lr = float(self.args.min_lr)
        self._warmup_steps = int(warmup_steps)
        self._terminal_anneal_epochs = int(
            getattr(self.args, "terminal_anneal_epochs", 0) or 0
        )
        if self._terminal_anneal_epochs < 0:
            raise ValueError("--terminal_anneal_epochs must be non-negative")
        if self._terminal_anneal_epochs >= int(self.args.num_epochs):
            raise ValueError(
                "--terminal_anneal_epochs must be smaller than --num_epochs")

        if self._lr_schedule == "plateau":
            # Linear warmup (applied in optimizer_step) then drop-on-plateau:
            # halve the LR after `patience` epochs without a val_loss
            # improvement, down to min_lr. Adapts to the actual convergence
            # curve, so it is robust when the run length / best schedule is not
            # known up front (unlike cosine, which is pinned to --num_epochs and
            # never reaches min_lr if you stop early). During warmup val_loss is
            # still improving, so the plateau scheduler does not fire — no
            # conflict with the manual warmup.
            plateau = ReduceLROnPlateau(
                optimizer, mode="min",
                factor=float(getattr(self.args, "plateau_factor", 0.5)),
                patience=int(getattr(self.args, "plateau_patience", 4)),
                min_lr=float(self.args.min_lr),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": plateau,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val_loss",
                },
            }

        min_ratio = float(self.args.min_lr) / max(float(self.args.start_lr), 1e-12)

        if self._lr_schedule == "step":
            # GGTF's schedule: multiply the LR by a constant factor every N
            # epochs, floored at min_lr, with no warmup. Their run goes 1e-3 to
            # 1e-6 in factor-0.1 steps.
            step_epochs = int(getattr(self.args, "lr_step_epochs", 4))
            step_factor = float(getattr(self.args, "lr_step_factor", 0.1))

            def step_lambda(step: int) -> float:
                epoch = step // max(steps_per_epoch, 1)
                return max(min_ratio, step_factor ** (epoch // max(step_epochs, 1)))

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": LambdaLR(optimizer, step_lambda),
                    "interval": "step",
                    "frequency": 1,
                },
            }

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # Manual linear warmup for the plateau schedule (the cosine schedule
        # handles warmup inside its LambdaLR, so it is left untouched).
        if getattr(self, "_lr_schedule", "cosine") == "plateau":
            ws = getattr(self, "_warmup_steps", 0)
            gs = self.trainer.global_step
            if ws and gs < ws:
                scale = float(gs + 1) / float(ws)
                for pg in optimizer.param_groups:
                    pg["lr"] = scale * self._base_lr
            else:
                terminal_epochs = getattr(self, "_terminal_anneal_epochs", 0)
                terminal_start = int(self.args.num_epochs) - terminal_epochs
                if terminal_epochs and epoch >= terminal_start:
                    # A deterministic upper bound on LR: never undo an earlier
                    # ReduceLROnPlateau drop, but guarantee a half-cosine
                    # refinement to min_lr by the final optimizer step.
                    epoch_fraction = float(batch_idx + 1) / max(
                        int(self._steps_per_epoch), 1)
                    progress = (
                        float(epoch) + epoch_fraction - terminal_start
                    ) / float(terminal_epochs)
                    progress = min(max(progress, 0.0), 1.0)
                    cap = self._min_lr + 0.5 * (
                        self._base_lr - self._min_lr
                    ) * (1.0 + math.cos(math.pi * progress))
                    for pg in optimizer.param_groups:
                        pg["lr"] = min(float(pg["lr"]), cap)
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self._finite_batches_since_optimizer_step = 0
