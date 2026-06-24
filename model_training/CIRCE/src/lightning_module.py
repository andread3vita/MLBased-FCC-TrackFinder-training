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
from torch.optim.lr_scheduler import LambdaLR

from src.model import (
    CGATrParquetModel, object_condensation_loss,
    _compute_batch_metrics_greedy, _seq_lens_to_batch, _compute_var_weight,
)


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
        mc_index_loss[is_secondary] = 0

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
        }

    # ---- training step ------------------------------------------------------
    def _dummy_ddp_step(self) -> torch.Tensor:
        """Zero-loss forward on a 2-hit dummy event. Keeps DDP allreduce balanced."""
        params = next(self.model.parameters())
        dummy = torch.zeros(2, 10, device=params.device, dtype=params.dtype)
        dummy[:, 3] = 1.0
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

        if (s["mc_index_loss"] != 0).sum() < 4:
            return (s["coords"].sum() + s["beta_val"].sum()) * 0.0

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
