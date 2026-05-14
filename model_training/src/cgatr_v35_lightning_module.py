"""v35 CGATr LightningModule (`v35lightning`).

Mirrors `src/train_cgatr_parquet.py` exactly under PyTorch Lightning:

  * Imports `CGATrParquetModel`, `object_condensation_loss`,
    `_compute_batch_metrics_greedy`, `_seq_lens_to_batch`, and
    `_compute_var_weight` from the v35 trainer via importlib so we
    never duplicate the model/loss definitions.
  * Same OC loss with `beta_suppress=0.1`, `qmin=0.1`, `var_weight=0.3`
    ramped linearly over `var_warmup_epochs` (epoch-based, 1-indexed
    to match v35).
  * Same EMA(0.999) over the full state_dict (BatchNorm buffers
    included). EMA is built lazily in `on_fit_start` so the shadow
    lives on the same device as the model. EMA weights are swapped
    in for validation and restored after.
  * Same AdamW(weight_decay=1e-4) + LambdaLR with linear warmup
    over `warmup_epochs` then half-cosine decay to `min_lr`.
  * No `len_balance`, no `focal_gamma`, no plateau scheduler — those
    are v37+ knobs and v35 doesn't carry them.

Per-epoch CSV (matching Run A's schema) is emitted by the trainer
driver (`src/train_cgatr_lightning_v35.py`) via a tiny callback.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from typing import Dict, List, Optional

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


# ---------------------------------------------------------------------------
# Re-export v35's model / loss / helpers so we never duplicate them. v35
# ships as a script (hyphen in the filename) so we have to load it via
# importlib.
# ---------------------------------------------------------------------------
_V35_PATH = os.path.join(os.path.dirname(__file__), "train_cgatr_parquet.py")
_spec = importlib.util.spec_from_file_location("_v35_train", _V35_PATH)
_v35 = importlib.util.module_from_spec(_spec)
sys.modules["_v35_train"] = _v35
_spec.loader.exec_module(_v35)

CGATrParquetModel = _v35.CGATrParquetModel
object_condensation_loss = _v35.object_condensation_loss
_compute_batch_metrics_greedy = _v35._compute_batch_metrics_greedy
_seq_lens_to_batch = _v35._seq_lens_to_batch
_compute_var_weight = _v35._compute_var_weight


class EMAShadow:
    """EMA over the full state_dict (parameters + BN running stats).

    Mirrors v35's `EMAModel` (`train_cgatr_parquet.py:60-79`):
    floating-point tensors are decayed; integer buffers (e.g.
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
    """Lightning wrapper around v35's `CGATrParquetModel` + OC loss."""

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
        # and grad accumulation. The explicit ctor arg is kept for unit
        # tests that build the module without a real trainer.
        self._steps_per_epoch = int(steps_per_epoch) if steps_per_epoch else 0

        self._ema: Optional[EMAShadow] = None
        self._ema_decay = float(getattr(args, "ema_decay", 0.0) or 0.0)
        # EMA bookkeeping: `_saved_train_state` is the live (pre-EMA)
        # weights stashed during validation; `_pending_ema_state` is an
        # EMA shadow loaded from a checkpoint BEFORE `on_fit_start` had
        # a chance to allocate `self._ema`.
        self._saved_train_state: Optional[Dict[str, torch.Tensor]] = None
        self._pending_ema_state: Optional[Dict[str, torch.Tensor]] = None

        # Per-epoch running stats. Train side uses manual accumulators
        # because `self.log(..., sync_dist=True)` deadlocks on divergent
        # ranks (see `training_step` for the skip paths).
        self._val_loss_sum: float = 0.0
        self._val_loss_n: int = 0
        self._val_metrics: List[Dict[str, float]] = []
        self._train_loss_sum: float = 0.0
        self._train_loss_n: int = 0

    def forward(self, features, seq_lens):
        return self.model(features, seq_lens)

    # ---- fit lifecycle ------------------------------------------------------
    def on_fit_start(self):
        # Allocate EMA lazily — Lightning moves the model to CUDA after
        # `__init__`, so building the shadow in the ctor would leave it
        # on CPU and force a per-step host↔device copy.
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
                f"[v35lightning] start  | num_epochs={self.args.num_epochs}, "
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
        mc_index_loss[is_secondary] = 0

        coords = output[:, :ed]
        if self.args.cosine_norm:
            coords = F.normalize(coords, dim=-1)
        beta_val = torch.sigmoid(output[:, ed])

        return {
            "coords": coords,
            "beta_val": beta_val,
            "mc_index": mc_index,
            "mc_index_loss": mc_index_loss,
            "is_secondary": is_secondary,
            "seq_lens": seq_lens,
            "batch_ids": batch_ids,
            "output": output,
        }

    # ---- training step ------------------------------------------------------
    def _dummy_ddp_step(self) -> torch.Tensor:
        """Zero-loss forward on a 2-hit dummy event.

        Used when the incoming batch is unusable (None sentinel). Keeps
        every DDP rank touching the same parameters so NCCL allreduce
        stays in lockstep — without this, ranks with real batches block
        on NCCL waiting for the skipping rank and the run deadlocks.
        Mirrors v35's `train_cgatr_parquet.py:436-444` block.
        """
        params = next(self.model.parameters())
        dummy = torch.zeros(2, 10, device=params.device, dtype=params.dtype)
        dummy[:, 3] = 1.0  # both hits as DC → exercise the DC branch
        out = self.model(dummy, [2])
        return out.sum() * 0.0

    def on_train_epoch_start(self):
        self._train_loss_sum = 0.0
        self._train_loss_n = 0

    def training_step(self, batch, batch_idx):
        # 1. Sentinel: dataset returned None for every event in the
        #    micro-batch. Dummy forward keeps DDP allreduce balanced.
        if batch is None:
            return self._dummy_ddp_step()

        s = self._shared_step(batch)
        n_events = len(s["seq_lens"])
        vw = _compute_var_weight(self.current_epoch + 1, self.args)  # 1-indexed

        # 2. Real batch but too few signal hits to compute OC loss. We
        #    anchor a zero-loss to the actual forward so allreduce still
        #    fires on this rank (don't accumulate it into the mean).
        if (s["mc_index_loss"] != 0).sum() < 4:
            return (s["coords"].sum() + s["beta_val"].sum()) * 0.0

        # 3. Healthy batch — run the OC loss.
        loss, comp = object_condensation_loss(
            coords=s["coords"],
            beta=s["beta_val"],
            mc_index=s["mc_index_loss"].long(),
            batch=s["batch_ids"].long(),
            qmin=self.args.qmin,
            attr_weight=self.args.attr_weight,
            repul_weight=self.args.repul_weight,
            fill_loss_weight=self.args.fill_loss_weight,
            use_average_cc_pos=self.args.use_average_cc_pos,
            beta_suppress_weight=self.args.beta_suppress_weight,
            var_weight=vw,
            return_components=True,
        )

        # 4. NaN/Inf guard: identical anchored-zero-loss recovery.
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            self.log("train/nan_skip", 1.0, on_step=True, on_epoch=False,
                     prog_bar=False, sync_dist=False, batch_size=n_events)
            return (s["coords"].sum() + s["beta_val"].sum()) * 0.0

        # Per-rank running mean; the DDP-correct global mean is computed
        # by an explicit all_reduce in `on_train_epoch_end`.
        self._train_loss_sum += float(loss.item())
        self._train_loss_n += 1

        # sync_dist=False is intentional everywhere on the training
        # step: ranks that hit a skip path above don't log here, so any
        # auto-sync at epoch end would deadlock NCCL. We aggregate
        # train_loss by hand in `on_train_epoch_end`.
        log_kwargs = dict(on_step=True, on_epoch=False,
                          sync_dist=False, batch_size=n_events)
        self.log("train/loss", loss.item(), prog_bar=True, **log_kwargs)
        self.log("train/L_att", float(comp["L_V_att"].item()), **log_kwargs)
        self.log("train/L_rep", float(comp["L_V_rep"].item()), **log_kwargs)
        self.log("train/L_var", float(comp["L_var"].item()), **log_kwargs)
        self.log("train/var_weight", vw, **log_kwargs)
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
            # Match v35: validate() does `continue`. In Lightning DDP val,
            # returning None is safe — Lightning doesn't allreduce per
            # validation step; we sync the reduced metrics at epoch end.
            return
        s = self._shared_step(batch)

        if (s["mc_index_loss"] != 0).sum() >= 4:
            loss = object_condensation_loss(
                coords=s["coords"],
                beta=s["beta_val"],
                mc_index=s["mc_index_loss"].long(),
                batch=s["batch_ids"].long(),
                qmin=self.args.qmin,
                attr_weight=self.args.attr_weight,
                repul_weight=self.args.repul_weight,
                var_weight=float(self.args.var_weight),
            )
            if not (torch.isnan(loss) or torch.isinf(loss)):
                self._val_loss_sum += float(loss.item())
                self._val_loss_n += 1

        ed = self.args.embed_dim
        m = _compute_batch_metrics_greedy(
            s["output"][:, :ed], s["output"][:, ed:ed + 1],
            s["mc_index"], s["is_secondary"], s["seq_lens"],
            tbeta=self.args.tbeta, td=self.args.td,
            cosine_norm=self.args.cosine_norm,
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

        # Validation runs the SAME code path on every rank (no skips),
        # so sync_dist=True is safe here and gives us a clean global mean.
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

        # train_loss reduction lives here (NOT in on_train_epoch_end)
        # because Lightning calls callback hooks before the
        # LightningModule's hook for the same event, so a self.log(...)
        # emitted from on_train_epoch_end would land in callback_metrics
        # AFTER _EpochCSVCallback.on_train_epoch_end has already read it.
        # on_validation_epoch_end runs strictly before on_train_epoch_end
        # for both module and callbacks, so the value is in place by the
        # time the CSV row is written. By this point all train batches
        # for the just-finished epoch have already run, so the totals
        # are stable. DDP all_reduce is explicit because some ranks
        # may skip per-step logging on sentinel / too-few-signal
        # batches and Lightning's auto on_epoch sync would deadlock.
        if not self.trainer.sanity_checking:
            device = next(self.model.parameters()).device
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
            # Mirror directly to callback_metrics so the CSV callback
            # sees the value without depending on Lightning's deferred
            # flush.
            self.trainer.callback_metrics["train_loss"] = torch.as_tensor(mean)
            self._train_loss_sum = 0.0
            self._train_loss_n = 0

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
        """Per-rank optimizer steps per epoch (drives warmup + cosine).

        Prefers `trainer.estimated_stepping_batches // num_epochs`,
        which is the only value that accounts for DDP sharding,
        `limit_train_batches`, and grad accumulation — the v35-Lightning
        smoke-run bug was using `len(train_loader)` (unsharded) and
        getting a schedule 4x off on a 4-GPU box. The explicit ctor arg
        is checked first only to keep `CGATrV35LightningModule()`
        testable without a real `Trainer`.
        """
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
        # Stash so on_fit_start / external code can inspect it.
        self._steps_per_epoch = steps_per_epoch

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.args.start_lr),
            weight_decay=float(getattr(self.args, "weight_decay", 1e-4)),
        )

        total_steps = self.args.num_epochs * steps_per_epoch
        warmup_steps = (
            self.args.warmup_steps
            if getattr(self.args, "warmup_steps", None) is not None
            else self.args.warmup_epochs * steps_per_epoch
        )

        min_ratio = float(self.args.min_lr) / max(float(self.args.start_lr), 1e-12)

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
