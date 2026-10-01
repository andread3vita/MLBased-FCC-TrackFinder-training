"""C-GATr FCC LightningModule (`cgatr_fcc`).

Imports CGATrParquetModel, object_condensation_loss, and helpers directly
from src.model (no importlib tricks needed — no hyphen in the filename).

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
from typing import Dict, List, Optional

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from src.model import (
    CGATrParquetModel, object_condensation_loss,
    _compute_batch_metrics_greedy, _seq_lens_to_batch, _compute_var_weight,
)


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
        self._val_loss_n: int = 0
        self._val_metrics: List[Dict[str, float]] = []
        self._train_loss_sum: float = 0.0
        self._train_loss_n: int = 0
        self._train_loss_sum_t: Optional[torch.Tensor] = None

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
        batch_ids = _seq_lens_to_batch(seq_lens, features.device)

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
            "noise_index": noise_index,
            "track_separation_weight": batch.get("track_separation_weight"),
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

    def training_step(self, batch, batch_idx):
        if batch is None:
            return self._dummy_ddp_step()

        s = self._shared_step(batch)
        n_events = len(s["seq_lens"])
        vw = _compute_var_weight(self.current_epoch + 1, self.args)  # 1-indexed

        if (s["mc_index_loss"] != s["noise_index"]).sum() < 4:
            return (s["coords"].sum() + s["beta_val"].sum()) * 0.0

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

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            self.log("train/nan_skip", 1.0, on_step=True, on_epoch=False,
                     prog_bar=False, sync_dist=False, batch_size=n_events)
            return (s["coords"].sum() + s["beta_val"].sum()) * 0.0

        loss_d = loss.detach()
        if self._train_loss_sum_t is None:
            self._train_loss_sum_t = loss_d.double().clone()
        else:
            self._train_loss_sum_t += loss_d.double()
        self._train_loss_n += 1

        log_kwargs = dict(on_step=True, on_epoch=False,
                          sync_dist=False, batch_size=n_events)
        self.log("train/loss", loss_d, prog_bar=True, **log_kwargs)
        self.log("train/L_att", comp["L_V_att"].detach(), **log_kwargs)
        self.log("train/L_rep", comp["L_V_rep"].detach(), **log_kwargs)
        self.log("train/L_beta_sig", comp["L_beta_sig"].detach(), **log_kwargs)
        self.log("train/L_beta_noise", comp["L_beta_noise"].detach(), **log_kwargs)
        self.log("train/L_beta_suppress",
                 comp["L_beta_suppress"].detach(), **log_kwargs)
        self.log("train/L_var", comp["L_var"].detach(), **log_kwargs)
        self.log("train/var_weight", float(vw), **log_kwargs)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"],
                 **log_kwargs)
        return loss

    # gradient clipping is handled by Trainer(gradient_clip_val=1.0)

    # ---- validation lifecycle ----------------------------------------------
    def on_validation_epoch_start(self):
        if self._ema is not None:
            self._saved_train_state = {
                k: v.detach().clone() for k, v in self.model.state_dict().items()
            }
            self.model.load_state_dict(self._ema.state_dict())
        self._val_loss_sum = 0.0
        self._val_loss_n = 0
        self._val_metrics = []

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return
        s = self._shared_step(batch)

        if (s["mc_index_loss"] != s["noise_index"]).sum() >= 4:
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
            if not (torch.isnan(loss) or torch.isinf(loss)):
                self._val_loss_sum += float(loss.item())
                self._val_loss_n += 1

        ed = self.args.embed_dim
        # `mc_index_loss`, not the raw column, so the metric scores the same target set the
        # loss was trained on. With no relabelling flags the two are identical on this dataset
        # (no secondaries, no unassociated hits), so this changes nothing retroactively -- it
        # is what makes `--min_target_hits` visible to strict50 rather than only to the loss.
        m = _compute_batch_metrics_greedy(
            s["output"][:, :ed], s["output"][:, ed:ed + 1],
            s["mc_index_loss"], s["is_secondary"], s["seq_lens"],
            tbeta=self.args.tbeta, td=self.args.td,
            cosine_norm=self.args.cosine_norm,
            noise_index=s["noise_index"],
        )
        self._val_metrics.append(m)

    def on_validation_epoch_end(self):
        avg_loss = self._val_loss_sum / max(self._val_loss_n, 1)
        if self._val_metrics:
            def _mean(key: str) -> float:
                return float(np.mean([m[key] for m in self._val_metrics]))
            avg_purity = _mean("purity")
            avg_eff = _mean("efficiency")
            avg_match_loose = _mean("match_rate")
            avg_match_strict50 = _mean("match_rate_strict50")
            avg_noise = _mean("noise_suppression")
        else:
            avg_purity = avg_eff = avg_match_loose = 0.0
            avg_match_strict50 = avg_noise = 0.0

        log_kwargs = dict(on_epoch=True, sync_dist=True, batch_size=1)
        self.log("val/loss", avg_loss, prog_bar=True, **log_kwargs)
        self.log("val_loss", avg_loss, **log_kwargs)
        self.log("val/purity", avg_purity, **log_kwargs)
        self.log("val/efficiency", avg_eff, **log_kwargs)
        self.log("val/match_rate", avg_match_loose, **log_kwargs)
        self.log("val/match_rate_strict50", avg_match_strict50, **log_kwargs)
        self.log("val/noise_supp", avg_noise, **log_kwargs)

        if self.trainer.is_global_zero:
            tag = ("[sanity]" if self.trainer.sanity_checking
                   else f"Epoch {self.current_epoch + 1}")
            print(
                f"  {tag} | Val Loss: {avg_loss:.4f} | "
                f"Purity: {avg_purity:.3f} | Efficiency: {avg_eff:.3f} | "
                f"Match: loose={avg_match_loose:.3f} "
                f"strict50={avg_match_strict50:.3f} | "
                f"Noise Supp: {avg_noise:.3f} ({self._val_loss_n} batches)",
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
                     batch_size=1)
            self.trainer.callback_metrics["train_loss"] = torch.as_tensor(mean)
            self._train_loss_sum = 0.0
            self._train_loss_n = 0
            self._train_loss_sum_t = None

    # ---- EMA + ckpt hooks ---------------------------------------------------
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self._ema is not None:
            self._ema.update(self.model)

    def on_save_checkpoint(self, checkpoint):
        if self._ema is not None:
            checkpoint["ema_state_dict"] = self._ema.state_dict()

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
        _use_fused = not _prec.startswith("16")
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
