"""GATr object-condensation tracking model with physics-aware validation."""

from __future__ import annotations

import ast
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import lightning as L
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from xformers.ops.fmha import BlockDiagonalMask

# gatr_v142 is vendored as a regular ``gatr`` package. Put that specific source
# tree first on sys.path and fail loudly if another installed GATr version was
# imported earlier in the process.
_GATR_V142_ROOT = Path(__file__).resolve().parents[1] / "gatr_v142"
if str(_GATR_V142_ROOT) not in sys.path:
    sys.path.insert(0, str(_GATR_V142_ROOT))

import gatr as _gatr_package
from gatr.interface import embed_point, embed_scalar, embed_translation
from gatr.layers.attention.config import SelfAttentionConfig
from gatr.layers.mlp.config import MLPConfig
from gatr.nets.gatr import GATr

_EXPECTED_GATR_PACKAGE = (_GATR_V142_ROOT / "gatr").resolve()
_LOADED_GATR_PACKAGE = Path(_gatr_package.__file__).resolve().parent
if _LOADED_GATR_PACKAGE != _EXPECTED_GATR_PACKAGE:
    raise ImportError(
        "The tracking model requires the vendored gatr_v142 package at "
        f"{_EXPECTED_GATR_PACKAGE}, but Python loaded {_LOADED_GATR_PACKAGE}."
    )

GATR_BACKEND = f"gatr_v142 (package version {_gatr_package.__version__})"
from src.layers.batch_operations import obtain_batch_numbers
from shared_training.circe_loss import object_condensation_loss
from shared_training.logging_contract import (
    log_sweep_metrics, log_training_metrics, log_validation_loss,
)
from shared_training.wandb_logger import (
    log_wandb_media, tracking_efficiency_media, wandb_html, wandb_image,
)
from shared_training.validation_media import validation_event_media
from shared_training.tracking_metrics import (
    TRACKING_COUNT_KEYS,
    TRACKING_DISPLACEMENT_BINS,
    TRACKING_PT_BINS,
    matching_comparison_binned_counts,
    matching_comparison_plot_series,
    parse_grid,
    run_operating_point_sweep,
    save_tracking_efficiency_displacement_plot,
    save_tracking_efficiency_pt_plot,
    save_sweep_results,
    tracking_metrics_from_counts,
)
from src.utils.lr_schedules import (
    NO_SCHEDULER_UNIT,
    REDUCE_ON_PLATEAU_UNIT,
    STEP_WARMUP_COSINE_UNIT,
    checkpoint_resume_metadata,
)
from src.utils.ema import EMAShadow
from src.utils.detector_features import (
    DEFAULT_LAYERS_PER_SUPERLAYER,
    DETECTOR_FEATURE_NAMES,
    validate_layers_per_superlayer,
)
from src.utils.checkpoint_weights import select_checkpoint_weights
from src.utils.track_weighting import (
    parse_pt_track_weight_config,
    pt_track_weights,
    weighted_mean,
)
from src.utils.validation import validation_output_name


class ExampleWrapper(L.LightningModule):
    helix_proxy_dim = 4
    lr_schedule_unit = "epoch-v1"

    @staticmethod
    def checkpoint_training_metadata(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            return {
                "contains_training_state": False,
                "learning_rates": [],
                "lr_scheduler": None,
                "lr_schedule_unit": None,
            }
        return checkpoint_resume_metadata(checkpoint)

    @staticmethod
    def checkpoint_contains_training_state(checkpoint_path):
        return ExampleWrapper.checkpoint_training_metadata(checkpoint_path)[
            "contains_training_state"
        ]

    def load_compatible_weights(self, checkpoint_path, weights_source="ema"):
        """Load compatible raw or EMA tensors without restoring training state."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        incoming = select_checkpoint_weights(checkpoint, weights_source)
        current = self.state_dict()
        compatible, skipped = {}, []
        for key, value in incoming.items():
            if key in current and current[key].shape == value.shape:
                compatible[key] = value
            else:
                skipped.append(key)
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        print(
            f"Loaded {len(compatible)} compatible {weights_source} tensors "
            f"from {checkpoint_path}; "
            f"initialized {len(missing)} new tensors and skipped {len(skipped)} changed tensors.",
            flush=True,
        )
        if unexpected:
            print(f"Unexpected checkpoint tensors: {unexpected}", flush=True)
        return self

    def __init__(self, args, dev=None):
        super().__init__()
        self.args = args
        scheduler_name = str(
            getattr(args, "lr_scheduler", "reduceplateau")
        ).lower()
        if scheduler_name == "none":
            self.lr_schedule_unit = NO_SCHEDULER_UNIT
        elif scheduler_name == "flat+decay":
            self.lr_schedule_unit = STEP_WARMUP_COSINE_UNIT
        elif scheduler_name == "reduceplateau":
            self.lr_schedule_unit = REDUCE_ON_PLATEAU_UNIT
        else:
            raise ValueError(
                "Lightning GATr supports lr_scheduler none, flat+decay, "
                "or reduceplateau"
            )
        self.embedding_dim = int(args.clustering_space_dim)
        self._steps_per_epoch = int(getattr(args, "steps_per_epoch", 0) or 0)
        self._base_lr = float(getattr(args, "start_lr", 4e-4))
        self._min_lr = float(getattr(args, "min_lr", 1e-5))
        self._warmup_steps = 0
        self._terminal_anneal_epochs = int(
            getattr(args, "terminal_anneal_epochs", 6) or 0
        )
        self.rejected_seed_policy = str(
            getattr(args, "rejected_seed_policy", "discard")
        ).lower()
        self.pt_track_weighting = bool(
            getattr(args, "pt_track_weighting", False)
        )
        (
            self.pt_track_weight_bin_edges,
            self.pt_track_weight_bin_weights,
        ) = parse_pt_track_weight_config(
            getattr(args, "pt_track_weight_bin_edges", "0.4,0.9,5.0"),
            getattr(
                args,
                "pt_track_weight_bin_weights",
                "1.5,1.2,0.75,2.0",
            ),
        )
        if self.pt_track_weighting and str(args.loss_type) != "hgcalimplementation":
            raise ValueError(
                "--pt-track-weighting currently requires "
                "--losstype hgcalimplementation"
            )
        self.use_detector_features = bool(
            getattr(args, "use_detector_features", False)
        )
        self.detector_scalar_dim = int(
            getattr(args, "detector_scalar_dim", 0)
        )
        expected_detector_scalar_dim = (
            len(DETECTOR_FEATURE_NAMES) if self.use_detector_features else 0
        )
        if self.detector_scalar_dim != expected_detector_scalar_dim:
            mode = "enabled" if self.use_detector_features else "disabled"
            raise ValueError(
                f"Detector features are {mode}, so detector_scalar_dim must "
                f"be {expected_detector_scalar_dim}; received "
                f"{self.detector_scalar_dim}."
            )
        self._use_helix_proxy = (
            float(getattr(args, "helix_loss_weight", 0.0)) > 0.0
        )
        self.output_dim = self.embedding_dim + 1 + self.helix_proxy_dim
        self.position_scale = float(getattr(args, "position_scale", 1000.0))
        self._val_cache = []
        self._val_loss_sum = 0.0
        self._val_loss_event_count = 0
        self._ema: Optional[EMAShadow] = None
        self._ema_decay = float(getattr(args, "ema_decay", 0.0))
        self._ema_restore_state: Optional[Dict[str, torch.Tensor]] = None
        self._pending_ema_state = None
        self.validation_output_tag = None
        self._validation_working_points = None
        self._saving_validation_sweep_checkpoint = False
        self._skipped_nonfinite_batches = 0
        self._finite_batches_since_optimizer_step = 0

        serializable = {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool, type(None)))
        }
        layers_per_superlayer = validate_layers_per_superlayer(
            getattr(args, "layers_per_superlayer", DEFAULT_LAYERS_PER_SUPERLAYER)
        )
        args.layers_per_superlayer = list(layers_per_superlayer)
        serializable["layers_per_superlayer"] = list(layers_per_superlayer)
        serializable["detector_feature_names"] = list(DETECTOR_FEATURE_NAMES)
        serializable["gatr_backend"] = GATR_BACKEND
        self.save_hyperparameters(serializable)

        checkpoint = (
            ["block"] if bool(getattr(args, "gradient_checkpointing", False)) else None
        )
        self.gatr = GATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=int(getattr(args, "hidden_mv_channels", 16)),
            in_s_channels=(
                self.detector_scalar_dim if self.use_detector_features else None
            ),
            out_s_channels=None,
            hidden_s_channels=int(getattr(args, "hidden_s_channels", 64)),
            num_blocks=int(getattr(args, "gatr_blocks", 10)),
            attention=SelfAttentionConfig(),
            mlp=MLPConfig(),
            checkpoint=checkpoint,
        )
        self.clustering = nn.Linear(16, self.embedding_dim, bias=False)
        self.beta = nn.Linear(16, 1)
        self.helix_proxy = nn.Linear(16, self.helix_proxy_dim)
        if not self._use_helix_proxy:
            self.helix_proxy.requires_grad_(False)

    def _detector_scalars(self, graph, dtype):
        """Return detector scalars unchanged, or ``None`` in geometry mode."""
        if not self.use_detector_features:
            return None
        if "scalar_features" not in graph.ndata:
            raise KeyError(
                "--use-detector-features requires "
                "graph.ndata['scalar_features']; use the detector data config"
            )
        values = graph.ndata["scalar_features"].to(dtype=dtype)
        expected_shape = (graph.num_nodes(), self.detector_scalar_dim)
        if tuple(values.shape) != expected_shape:
            raise ValueError(
                "Expected detector scalars with shape "
                f"{expected_shape}, received {tuple(values.shape)}"
            )
        return values

    def _attention_order(self, graph, positions):
        """Optionally partition each event into phi sectors for lower O(N^2) cost."""
        batch_numbers = obtain_batch_numbers(graph).long()
        sectors = int(getattr(self.args, "attention_phi_sectors", 1))
        if sectors <= 1:
            return None, torch.bincount(batch_numbers).tolist()
        phi = torch.atan2(positions[:, 1], positions[:, 0])
        sector = torch.floor((phi + torch.pi) * sectors / (2 * torch.pi)).long()
        sector.clamp_(0, sectors - 1)
        group = batch_numbers * sectors + sector
        order = torch.argsort(group, stable=True)
        lengths = torch.bincount(group[order])
        return order, lengths[lengths > 0].tolist()

    @staticmethod
    def _join_reference(multivectors, sequence_lengths):
        """Construct one data-derived join reference per attention sequence.

        Upstream GATr's ``join_reference="data"`` cannot infer event boundaries
        from an xFormers block-diagonal mask. Supplying this tensor preserves the
        per-event behavior of the previous tracking fork (or per-sector behavior
        when phi-sector attention is enabled).
        """
        references = []
        start = 0
        for length in sequence_lengths:
            end = start + int(length)
            sequence_mean = multivectors[start:end].mean(dim=0, keepdim=True)
            references.append(
                sequence_mean.expand(end - start, *sequence_mean.shape[1:])
            )
            start = end
        if start != multivectors.shape[0]:
            raise RuntimeError(
                "Attention sequence lengths do not cover all input tokens: "
                f"covered {start}, expected {multivectors.shape[0]}"
            )
        return torch.cat(references, dim=0)

    def forward(self, graph, input_tensor=None):
        # Lightning owns the model dtype. Align floating graph features with the
        # model at the boundary instead of hardcoding a particular precision in
        # the data pipeline.
        model_dtype = self.clustering.weight.dtype
        positions = graph.ndata["pos_hits_xyz"].to(dtype=model_dtype)
        hit_type = graph.ndata["hit_type"].reshape(-1, 1).to(dtype=model_dtype)
        vector = graph.ndata["vector"].to(dtype=model_dtype)
        positions_scaled = positions / self.position_scale
        vector_scaled = vector / self.position_scale
        multivectors = (
            embed_point(positions_scaled)
            + embed_scalar(hit_type)
            + embed_translation(vector_scaled)
        ).unsqueeze(-2)
        scalars = self._detector_scalars(graph, model_dtype)
        order, sequence_lengths = self._attention_order(graph, positions)
        if order is not None:
            multivectors = multivectors[order]
            if scalars is not None:
                scalars = scalars[order]
        mask = BlockDiagonalMask.from_seqlens(sequence_lengths)
        join_reference = self._join_reference(multivectors, sequence_lengths)
        embedded, _ = self.gatr(
            multivectors,
            scalars=scalars,
            attention_mask=mask,
            join_reference=join_reference,
        )
        latent = embedded[:, 0, :]
        if order is not None:
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(order.numel(), device=order.device)
            latent = latent[inverse]
        if self._use_helix_proxy:
            helix_prediction = self.helix_proxy(latent)
        else:
            helix_prediction = latent.new_zeros(
                (latent.shape[0], self.helix_proxy_dim)
            )
        return torch.cat(
            (self.clustering(latent), self.beta(latent), helix_prediction), dim=1
        )

    def _split_output(self, output):
        coords = output[:, : self.embedding_dim]
        beta_logits = output[:, self.embedding_dim]
        auxiliary = output[:, self.embedding_dim + 1 :]
        return coords, beta_logits, auxiliary

    def _variance_weight(self):
        target = float(getattr(self.args, "var_weight", 0.0))
        warmup = int(getattr(self.args, "var_warmup_epochs", 0))
        if warmup <= 0:
            return target
        return target * min(1.0, max(0.0, self.current_epoch / warmup))

    def _validation_variance_weight(self):
        """Use CIRCE's final variance weight for every validation epoch."""
        return float(getattr(self.args, "var_weight", 0.0))

    def _helix_proxy_loss(self, graph, y, prediction):
        """Auxiliary inverse-pT/direction regression using available particle labels."""
        weight = float(getattr(self.args, "helix_loss_weight", 0.0))
        if weight <= 0 or prediction.numel() == 0:
            return prediction.sum() * 0.0

        # Match nodes to truth particles entirely on the accelerator.  The old
        # implementation built a Python dictionary for every event and called
        # .tolist()/.item() for every signal node, forcing repeated GPU-to-CPU
        # synchronization in every training batch.
        node_counts = graph.batch_num_nodes().to(device=prediction.device)
        batch_numbers = torch.repeat_interleave(
            torch.arange(node_counts.numel(), device=prediction.device),
            node_counts,
        )
        particle_ids = graph.ndata["particle_number_nomap"].reshape(-1)
        signal = graph.ndata["particle_number"].reshape(-1) > 0
        signal_indices = torch.nonzero(signal, as_tuple=False).flatten()
        if y.shape[0] == 0 or signal_indices.numel() == 0:
            return prediction.sum() * 0.0

        truth_event_ids = y[:, -1].long()
        truth_particle_ids = y[:, 4].long()
        signal_particle_ids = particle_ids[signal_indices].long()

        # A dynamic stride gives each (event, particle ID) pair a unique int64
        # key without assuming that particle IDs are consecutive or positive.
        minimum_particle_id = torch.minimum(
            truth_particle_ids.min(), signal_particle_ids.min()
        )
        maximum_particle_id = torch.maximum(
            truth_particle_ids.max(), signal_particle_ids.max()
        )
        key_stride = maximum_particle_id - minimum_particle_id + 1
        truth_keys = (
            truth_event_ids * key_stride
            + truth_particle_ids
            - minimum_particle_id
        )
        signal_keys = (
            batch_numbers[signal_indices] * key_stride
            + signal_particle_ids
            - minimum_particle_id
        )

        sorted_truth_keys, truth_order = torch.sort(truth_keys)
        match_positions = torch.searchsorted(sorted_truth_keys, signal_keys)
        safe_positions = match_positions.clamp(max=sorted_truth_keys.numel() - 1)
        matched = (match_positions < sorted_truth_keys.numel()) & (
            sorted_truth_keys[safe_positions] == signal_keys
        )
        matched_node_indices = signal_indices[matched]
        matched_truth_positions = safe_positions[matched]
        particle_rows = y[truth_order[matched_truth_positions]]

        theta = particle_rows[:, 0]
        phi = particle_rows[:, 1]
        pt = particle_rows[:, 6].abs().clamp(min=1e-3)
        targets = torch.stack(
            (
                torch.log1p(1.0 / pt),
                torch.sin(phi),
                torch.cos(phi),
                torch.reciprocal(torch.tan(theta)).clamp(-10, 10) / 10.0,
            ),
            dim=1,
        ).to(dtype=prediction.dtype)

        # Ignore malformed truth rows rather than allowing one NaN direction
        # or momentum value to invalidate the complete mini-batch.
        finite_targets = torch.isfinite(targets).all(dim=1)
        matched_node_indices = matched_node_indices[finite_targets]
        matched_truth_positions = matched_truth_positions[finite_targets]
        targets = targets[finite_targets]
        if matched_node_indices.numel() == 0:
            return prediction.sum() * 0.0

        if self.pt_track_weighting:
            elementwise_loss = torch.nn.functional.smooth_l1_loss(
                prediction[matched_node_indices], targets, reduction="none"
            )
            loss_per_hit = elementwise_loss.mean(dim=1)
            loss_sum_per_track = torch.zeros(
                y.shape[0], dtype=loss_per_hit.dtype, device=loss_per_hit.device
            )
            hit_count_per_track = torch.zeros_like(loss_sum_per_track)
            loss_sum_per_track.scatter_add_(
                0, matched_truth_positions, loss_per_hit
            )
            hit_count_per_track.scatter_add_(
                0,
                matched_truth_positions,
                torch.ones_like(loss_per_hit),
            )
            represented = hit_count_per_track > 0
            loss_per_track = (
                loss_sum_per_track[represented]
                / hit_count_per_track[represented]
            )
            sorted_pt = y[truth_order, 6].to(device=prediction.device)
            track_weights = pt_track_weights(
                sorted_pt[represented],
                self.pt_track_weight_bin_edges,
                self.pt_track_weight_bin_weights,
                dtype=loss_per_track.dtype,
            )
            return weighted_mean(loss_per_track, track_weights)

        loss_sum = torch.nn.functional.smooth_l1_loss(
            prediction[matched_node_indices], targets, reduction="sum"
        )
        # Match PyTorch's original reduction="mean": average over both the
        # matched hits and the four helix target components.
        normalizer = max(
            matched_node_indices.numel() * prediction.shape[1],
            1,
        )
        return loss_sum / normalizer

    def _shared_step(self, batch, stage):
        graph, _ = batch
        output = self(graph)
        coords, beta_logits, _ = self._split_output(output)
        variance_weight = (
            self._validation_variance_weight()
            if stage == "val"
            else self._variance_weight()
        )
        batch_ids = torch.repeat_interleave(
            torch.arange(
                len(graph.batch_num_nodes()), device=output.device,
                dtype=torch.long,
            ),
            graph.batch_num_nodes().to(output.device),
        )
        loss, components = object_condensation_loss(
            coords=coords.float(),
            beta=torch.sigmoid(beta_logits.float()),
            mc_index=graph.ndata["particle_number"].view(-1).long(),
            batch=batch_ids,
            noise_index=0,
            qmin=float(self.args.qmin),
            attr_weight=float(self.args.L_attractive_weight),
            repul_weight=float(self.args.L_repulsive_weight),
            beta_suppress_weight=float(getattr(self.args, "beta_suppress_weight", 0.0)),
            var_weight=variance_weight,
            return_components=True,
            oc_mode="paper_hinge",
        )
        batch_size = len(graph.batch_num_nodes())
        if stage == "train":
            loss_metrics = log_training_metrics(
                self,
                loss,
                components,
                self.args.L_attractive_weight,
                self.args.L_repulsive_weight,
                variance_weight,
                batch_size,
            )
            self._latest_loss_metrics = {
                metric_name: (
                    metric_value.detach()
                    if torch.is_tensor(metric_value)
                    else torch.as_tensor(metric_value, device=loss.device)
                )
                for metric_name, metric_value in loss_metrics.items()
            }
        return loss, output

    def _nonfinite_batch_details(self, graph, output):
        """Describe bad inputs/outputs only after a non-finite batch is found."""
        details = []
        offset = 0
        fields = (
            "pos_hits_xyz",
            "vector",
            "hit_type",
        )
        if self.use_detector_features:
            fields += ("scalar_features",)
        for event_slot, node_count in enumerate(graph.batch_num_nodes().tolist()):
            end = offset + int(node_count)
            field_counts = []
            for field in fields:
                if field in graph.ndata:
                    values = graph.ndata[field][offset:end]
                    count = int((~torch.isfinite(values)).sum().cpu().item())
                    field_counts.append(f"{field}={count}")
            output_count = int(
                (~torch.isfinite(output[offset:end])).sum().cpu().item()
            )
            file_number = int(graph.ndata["fileNumber"][offset].cpu().item())
            event_number = int(graph.ndata["eventNumber"][offset].cpu().item())
            details.append(
                f"slot={event_slot},file={file_number},event={event_number},"
                f"nodes={node_count},nonfinite_output={output_count},"
                + ",".join(field_counts)
            )
            offset = end
        return " | ".join(details)

    def _zero_loss_for_skipped_batch(self, reference):
        """Return a finite zero connected to every trainable DDP parameter."""
        zero_loss = reference.detach().new_zeros(())
        for parameter in self.parameters():
            if parameter.requires_grad and parameter.numel() > 0:
                zero_loss = zero_loss + parameter.reshape(-1)[0] * 0.0
        return zero_loss

    def training_step(self, batch, batch_idx):
        loss, output = self._shared_step(batch, "train")
        local_bad = (~torch.isfinite(loss)) | (~torch.isfinite(output).all())
        global_bad = local_bad.to(dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_bad, op=dist.ReduceOp.MAX)

        if bool(global_bad.cpu().item()):
            trainable_parameters = [
                parameter
                for parameter in self.parameters()
                if parameter.requires_grad
            ]
            parameters_finite = torch.stack(
                [
                    torch.isfinite(parameter.detach()).all()
                    for parameter in trainable_parameters
                ]
            ).all().to(dtype=torch.int32, device=loss.device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(parameters_finite, op=dist.ReduceOp.MIN)
            if not bool(parameters_finite.cpu().item()):
                raise FloatingPointError(
                    "Model parameters became non-finite; refusing to skip batches "
                    "because the checkpoint/model state is already corrupted."
                )

            if bool(local_bad.cpu().item()):
                component_values = ", ".join(
                    f"{name}={value.float().cpu().item():.6g}"
                    for name, value in self._latest_loss_metrics.items()
                )
                print(
                    f"Skipping non-finite training batch {batch_idx} on rank "
                    f"{self.global_rank}; {component_values}; "
                    f"{self._nonfinite_batch_details(batch[0], output)}",
                    flush=True,
                )

            self._skipped_nonfinite_batches += 1
            if self.trainer.is_global_zero:
                print(
                    f"All ranks skipped training batch {batch_idx}; total skipped "
                    f"batches={self._skipped_nonfinite_batches}",
                    flush=True,
                )
            # Lightning does not allow ``None`` from training_step under DDP.
            # A parameter-connected zero runs a synchronized zero backward pass,
            # so this batch contributes no gradients on any rank.  Connecting
            # every trainable parameter also keeps DDP happy when unused-parameter
            # detection is disabled.
            return self._zero_loss_for_skipped_batch(loss)

        self._finite_batches_since_optimizer_step += 1
        return loss

    def on_before_optimizer_step(self, optimizer):
        # With gradient accumulation, retain gradients from any finite batches in
        # the window.  If the complete window was non-finite, clearing gradients
        # to None prevents Adam/AdamW momentum and weight decay from changing the
        # parameters on what should be a genuinely skipped update.
        if self._finite_batches_since_optimizer_step == 0:
            optimizer.zero_grad(set_to_none=True)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Run one optimizer update, then advance EMA exactly once.

        Lightning does not call this hook for the intermediate mini-batches in a
        gradient-accumulation window.  Keeping the EMA update here therefore
        makes ``ema_decay`` an optimizer-step decay, independently of
        ``accumulate_grad_batches``.
        """
        if str(getattr(self.args, "lr_scheduler", "")).lower() == "reduceplateau":
            global_step = self.trainer.global_step
            if self._warmup_steps and global_step < self._warmup_steps:
                scale = float(global_step + 1) / float(self._warmup_steps)
                for group in optimizer.param_groups:
                    group["lr"] = scale * self._base_lr
            else:
                terminal_start = int(self.args.num_epochs) - self._terminal_anneal_epochs
                if self._terminal_anneal_epochs and epoch >= terminal_start:
                    epoch_fraction = float(batch_idx + 1) / max(
                        self._steps_per_epoch, 1
                    )
                    progress = (
                        float(epoch) + epoch_fraction - terminal_start
                    ) / float(self._terminal_anneal_epochs)
                    progress = min(max(progress, 0.0), 1.0)
                    cap = self._min_lr + 0.5 * (
                        self._base_lr - self._min_lr
                    ) * (1.0 + math.cos(math.pi * progress))
                    for group in optimizer.param_groups:
                        group["lr"] = min(float(group["lr"]), cap)
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if self._ema is not None and self._finite_batches_since_optimizer_step > 0:
            self._ema.update(self)
        self._finite_batches_since_optimizer_step = 0

    def on_validation_epoch_start(self):
        self._val_cache = []
        self._val_loss_sum = 0.0
        self._val_loss_event_count = 0
        self._validation_working_points = None
        self._initialize_ema_if_needed()
        if self._ema is not None:
            self._ema_restore_state = {
                key: value.detach().clone() for key, value in self.state_dict().items()
            }
            self.load_state_dict(self._ema.state_dict(), strict=False)

    def _cache_validation_output(self, graph, y, output):
        """Cache every event processed by validation on CPU.

        The operating-point sweep takes a configured prefix of this cache;
        the pT-dependent efficiency uses the complete cache.
        """
        coords, beta_logits, _ = self._split_output(output)
        truth = graph.ndata["particle_number"].detach().cpu().numpy()
        truth_particle_id = (
            graph.ndata["particle_number_nomap"].detach().cpu().numpy()
        )
        mc_particle_id = (
            graph.ndata["particle_number_nomap_original"].detach().cpu().numpy()
        )
        coords = coords.detach().float().cpu().numpy()
        beta = torch.sigmoid(beta_logits).detach().float().cpu().numpy()
        particle_rows = y.detach().cpu().numpy()
        particle_batch = particle_rows[:, -1].astype(np.int64, copy=False)

        # Particle feature columns follow config_tracking_parquet.yaml:
        # theta=0, particle ID=4, pT=6, generator status=7, and production
        # vertex x/y/z=9/10/11. The collator adds the event slot last.
        offset = 0
        for event_slot, length in enumerate(graph.batch_num_nodes().tolist()):
            end = offset + int(length)
            event_truth = truth[offset:end]
            event_original_id = truth_particle_id[offset:end]
            rows = particle_rows[particle_batch == event_slot]
            particles_by_original_id = {
                int(round(float(row[4]))): {
                    "theta": float(row[0]),
                    "pt": float(row[6]),
                    "gen_status": float(row[7]),
                    # Transverse production displacement in mm. The complete
                    # vertex remains available in the Parquet particle fields.
                    "displacement": float(np.hypot(row[9], row[10])),
                }
                for row in rows
            }
            particle_info = {}
            for truth_id in np.unique(event_truth[event_truth > 0]):
                original_ids = np.unique(
                    event_original_id[
                        (event_truth == truth_id) & (event_original_id >= 0)
                    ]
                )
                if original_ids.size != 1:
                    continue
                original_id = int(round(float(original_ids[0])))
                if original_id in particles_by_original_id:
                    particle_info[int(truth_id)] = particles_by_original_id[original_id]

            self._val_cache.append(
                {
                    # Keep detector coordinates for the interactive validation
                    # display.  ``coords`` above are the learned embedding
                    # coordinates, not the detector hit positions.
                    "positions": graph.ndata["pos_hits_xyz"]
                    .detach()
                    .cpu()
                    .numpy()[offset:end],
                    "coords": coords[offset:end],
                    "beta": beta[offset:end],
                    "truth": event_truth,
                    "mc_particle_id": mc_particle_id[offset:end],
                    "particle_info": particle_info,
                }
            )
            offset = end

    @staticmethod
    def _validation_scatter_figure(
        points, labels, colour_title, title, axis_titles
    ):
        """Create an interactive 2D or 3D one-trace-per-particle figure."""
        import plotly.graph_objects as go

        points = np.asarray(points)
        labels = np.asarray(labels, dtype=np.int64)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError(
                "Validation plotting requires an array with two or three columns; "
                f"received shape {points.shape}."
            )
        if points.shape[0] != labels.shape[0]:
            raise ValueError(
                "Validation plotting received different numbers of points and labels: "
                f"{points.shape[0]} and {labels.shape[0]}."
            )

        traces = []
        # A separate trace for every label makes each particle independently
        # hideable through the Plotly/W&B legend.
        for label in sorted(np.unique(labels).tolist()):
            mask = labels == label
            label_text = "unassigned" if label < 0 else str(int(label))
            if points.shape[1] == 2:
                trace = go.Scatter(
                    x=points[mask, 0],
                    y=points[mask, 1],
                    mode="markers",
                    name=label_text,
                    marker={"size": 5},
                    hovertemplate=(
                        f"{colour_title}: {label_text}<br>"
                        f"{axis_titles[0]}=%{{x:.3f}}<br>"
                        f"{axis_titles[1]}=%{{y:.3f}}<extra></extra>"
                    ),
                )
            else:
                trace = go.Scatter3d(
                    x=points[mask, 0],
                    y=points[mask, 1],
                    z=points[mask, 2],
                    mode="markers",
                    name=label_text,
                    marker={"size": 3},
                    hovertemplate=(
                        f"{colour_title}: {label_text}<br>"
                        f"{axis_titles[0]}=%{{x:.3f}}<br>"
                        f"{axis_titles[1]}=%{{y:.3f}}<br>"
                        f"{axis_titles[2]}=%{{z:.3f}}<extra></extra>"
                    ),
                )
            traces.append(trace)

        fig = go.Figure(traces)
        layout = {
            "title": title,
            "legend": {
                "title": {"text": colour_title},
                "itemsizing": "constant",
            },
            "margin": {"l": 0, "r": 0, "b": 0, "t": 45},
        }
        if points.shape[1] == 2:
            layout.update(
                xaxis={"title": axis_titles[0]},
                yaxis={"title": axis_titles[1]},
            )
        else:
            layout["scene"] = {
                "xaxis_title": axis_titles[0],
                "yaxis_title": axis_titles[1],
                "zaxis_title": axis_titles[2],
            }
        fig.update_layout(**layout)
        return fig

    @staticmethod
    def _embedding_plot_coordinates(coords):
        """Return direct 2D/3D coordinates or a deterministic three-component PCA."""
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(
                "Embedding visualization requires at least two dimensions; "
                f"received shape {coords.shape}."
            )
        embedding_dim = coords.shape[1]
        if embedding_dim == 2:
            return coords, ("embedding 0", "embedding 1"), "2D embedding"
        if embedding_dim == 3:
            return (
                coords,
                ("embedding 0", "embedding 1", "embedding 2"),
                "3D embedding",
            )

        centered = coords - np.mean(coords, axis=0, keepdims=True)
        _, _, components = np.linalg.svd(centered, full_matrices=False)
        components = components[:3].copy()
        # PCA component signs are mathematically arbitrary.  Anchor each sign
        # to its largest-magnitude loading so repeated plots do not flip merely
        # because of an SVD sign convention.
        for component in components:
            pivot = int(np.argmax(np.abs(component)))
            if component[pivot] < 0:
                component *= -1
        projected = centered @ components.T
        if projected.shape[1] < 3:
            projected = np.pad(
                projected,
                ((0, 0), (0, 3 - projected.shape[1])),
                mode="constant",
            )
        return (
            projected,
            ("PC1", "PC2", "PC3"),
            f"PCA of {embedding_dim}D embedding",
        )

    def _first_validation_event_media(self, event, working_point, output_dir=None):
        """Build detector- and embedding-space media for the first event."""
        html_payloads = validation_event_media(
            event,
            working_point,
            rejected_seed_policy=self.rejected_seed_policy,
            output_dir=output_dir,
            include_embedding=True,
        )
        return {key: wandb_html(html) for key, html in html_payloads.items()}

    def _local_validation_sweep_events(self):
        """Take this rank's share of the configured global sweep limit."""
        max_events = int(getattr(self.args, "validation_sweep_max_events", 100))
        world_size = max(int(self.trainer.world_size), 1)
        rank = int(self.trainer.global_rank)
        events_per_rank, remainder = divmod(max_events, world_size)
        per_rank_limit = events_per_rank + int(rank < remainder)
        return self._val_cache[:per_rank_limit]

    def validation_step(self, batch, batch_idx):
        loss, output = self._shared_step(batch, "val")
        if torch.isfinite(loss):
            n_events = len(batch[0].batch_num_nodes())
            self._val_loss_sum += float(loss.item()) * n_events
            self._val_loss_event_count += n_events
        if not self.trainer.sanity_checking:
            self._cache_validation_output(batch[0], batch[1], output)
        return loss

    def on_validation_epoch_end(self):
        reduction_device = next(self.parameters()).device
        loss_stats = torch.tensor(
            [self._val_loss_sum, self._val_loss_event_count],
            dtype=torch.float64,
            device=reduction_device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
        global_loss_sum, global_event_count = loss_stats.tolist()
        validation_loss = global_loss_sum / max(global_event_count, 1.0)
        log_validation_loss(self, validation_loss, batch_size=1)

        if not self.trainer.sanity_checking:
            # Collect every image/HTML object and send one W&B history row per
            # validation pass.  Separate logger.log_image() calls each commit a
            # row and advance W&B's step even though no training occurred.
            wandb_media = {}
            output_dir = (
                Path(self.args.model_prefix)
                / "validation_sweeps"
                / validation_output_name(
                    self.validation_output_tag, self.current_epoch
                )
            )
            tbetas = parse_grid(self.args.sweep_tbeta_grid, float)
            tds = parse_grid(self.args.sweep_td_grid, float)
            min_hits = parse_grid(self.args.sweep_min_hits_grid, int)
            local_sweep_events = self._local_validation_sweep_events()
            local_rows = run_operating_point_sweep(
                local_sweep_events,
                tbetas,
                tds,
                min_hits,
                metric=self.args.sweep_match_metric,
                truth_min_hits=int(self.args.sweep_truth_min_hits),
                rejected_seed_policy=self.rejected_seed_policy,
            )

            # Each rank evaluates only its local events.  Only the additive raw
            # counts are reduced, so global rates retain their correct event
            # weighting without gathering embeddings or predictions to rank 0.
            count_tensor = torch.tensor(
                [
                    [row[key] for key in TRACKING_COUNT_KEYS]
                    for row in local_rows
                ],
                dtype=torch.int64,
                device=reduction_device,
            )
            event_count = torch.tensor(
                len(local_sweep_events), dtype=torch.int64, device=reduction_device
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(event_count, op=dist.ReduceOp.SUM)

            working_points = None
            if self.trainer.is_global_zero and event_count.item() > 0:
                rows = []
                for local_row, aggregate_counts in zip(
                    local_rows, count_tensor.cpu().tolist()
                ):
                    totals = dict(zip(TRACKING_COUNT_KEYS, aggregate_counts))
                    rows.append(
                        tracking_metrics_from_counts(
                            local_row["tbeta"],
                            local_row["td"],
                            local_row["min_hits"],
                            totals,
                        )
                    )
                total_events = int(event_count.item())
                metadata = {
                    "epoch": int(self.current_epoch),
                    "validation_phase": (
                        self.validation_output_tag or "training"
                    ),
                    "events": total_events,
                    "distributed_ranks": int(self.trainer.world_size),
                    "embedding_dim": self.embedding_dim,
                    "qmin": float(self.args.qmin),
                    "attractive_weight": float(self.args.L_attractive_weight),
                    "repulsive_weight": float(self.args.L_repulsive_weight),
                    "beta_suppress_weight": float(self.args.beta_suppress_weight),
                    "beta_second_weight": float(self.args.beta_second_weight),
                    "variance_weight": self._validation_variance_weight(),
                    "helix_loss_weight": float(self.args.helix_loss_weight),
                    "hard_negative_weight": float(self.args.hard_negative_weight),
                    "pt_track_weighting": self.pt_track_weighting,
                    "pt_track_weight_bin_edges": list(
                        self.pt_track_weight_bin_edges
                    ),
                    "pt_track_weight_bin_weights": list(
                        self.pt_track_weight_bin_weights
                    ),
                    "rejected_seed_policy": self.rejected_seed_policy,
                    "matching": self.args.sweep_match_metric,
                    "working_point_min_hits": int(self.args.sweep_truth_min_hits),
                }
                working_points = save_sweep_results(
                    rows,
                    str(output_dir),
                    metadata,
                    fixed_min_hits=int(self.args.sweep_truth_min_hits),
                )
                max_efficiency_wp = working_points["max_efficiency"]
                pareto_f1_wp = working_points["pareto_f1"]
                print(
                    f"Distributed validation sweep over {total_events} events selected "
                    "maximum-efficiency WP: "
                    f"tbeta={max_efficiency_wp['tbeta']:g}, "
                    f"td={max_efficiency_wp['td']:g}, "
                    f"min_hits={max_efficiency_wp['min_hits']}, "
                    f"eff={max_efficiency_wp['efficiency']:.4f}, "
                    f"fake={max_efficiency_wp['fake_rate']:.4f}, "
                    f"F1={max_efficiency_wp['f1']:.4f}; "
                    "Pareto maximum-F1 WP: "
                    f"tbeta={pareto_f1_wp['tbeta']:g}, "
                    f"td={pareto_f1_wp['td']:g}, "
                    f"min_hits={pareto_f1_wp['min_hits']}, "
                    f"eff={pareto_f1_wp['efficiency']:.4f}, "
                    f"fake={pareto_f1_wp['fake_rate']:.4f}, "
                    f"F1={pareto_f1_wp['f1']:.4f}",
                    flush=True,
                )
                if hasattr(self.logger, "log_image"):
                    images = [output_dir / "efficiency_fake_pareto.png"]
                    images.extend(sorted(output_dir.glob("sweep_min_hits_*.png")))
                    images = [str(path) for path in images if path.is_file()]
                    for image_path in images:
                        wandb_media[
                            (
                                "plots/operating_point_sweep/"
                                f"{Path(image_path).stem}"
                            )
                        ] = wandb_image(image_path)
            if dist.is_available() and dist.is_initialized():
                holder = [working_points]
                dist.broadcast_object_list(holder, src=0)
                working_points = holder[0]
            if working_points is not None:
                max_efficiency_wp = working_points["max_efficiency"]
                pareto_f1_wp = working_points["pareto_f1"]
                # Keep the exact sweep selections available until the two
                # validation-end checkpoints have been serialized.
                self._validation_working_points = {
                    name: dict(working_point)
                    for name, working_point in working_points.items()
                }

                # Rank zero owns the first validation shard/event. Log detector-
                # and embedding-space views after every validation epoch using
                # the selected Pareto operating point for reconstruction labels.
                if (
                    self.trainer.is_global_zero
                    and self._val_cache
                    and hasattr(self.logger, "log_image")
                ):
                    wandb_media.update(
                        self._first_validation_event_media(
                            self._val_cache[0], pareto_f1_wp, output_dir=output_dir
                        )
                    )

                log_sweep_metrics(self, working_points)

                # These logger-hidden aliases are retained for the scheduler,
                # progress bar, and checkpoint filename/monitor.
                internal_metrics = {
                    "val_pareto_f1": pareto_f1_wp["f1"],
                    "val_pareto_fake_rate": pareto_f1_wp["fake_rate"],
                    "val_pareto_tracking_efficiency": pareto_f1_wp["efficiency"],
                    "val_pareto_tbeta": pareto_f1_wp["tbeta"],
                    "val_pareto_td": pareto_f1_wp["td"],
                    "val_pareto_min_hits": pareto_f1_wp["min_hits"],
                    "val_max_tracking_efficiency": max_efficiency_wp["efficiency"],
                    "val_max_efficiency_fake_rate": max_efficiency_wp["fake_rate"],
                    "val_max_efficiency_f1": max_efficiency_wp["f1"],
                    "val_max_efficiency_tbeta": max_efficiency_wp["tbeta"],
                    "val_max_efficiency_td": max_efficiency_wp["td"],
                    "val_max_efficiency_min_hits": max_efficiency_wp["min_hits"],
                    # Backward-compatible aliases used by existing dashboards.
                    "val_fake_rate": pareto_f1_wp["fake_rate"],
                    "val_tracking_efficiency": pareto_f1_wp["efficiency"],
                }
                progress_bar_metrics = {
                    "val_pareto_f1",
                    "val_pareto_fake_rate",
                    "val_pareto_tracking_efficiency",
                    "val_max_tracking_efficiency",
                }
                for metric_name, metric_value in internal_metrics.items():
                    self.log(
                        metric_name,
                        float(metric_value),
                        on_epoch=True,
                        sync_dist=True,
                        prog_bar=metric_name in progress_bar_metrics,
                        logger=False,
                        batch_size=1,
                    )

                # Compare the thresholded double-majority association with a
                # threshold-free global Hungarian one-to-one assignment at the
                # same clustering working points.
                comparison_order, comparison_rows, missing_info = (
                    matching_comparison_binned_counts(
                        self._val_cache,
                        working_points,
                        {
                            "pt": TRACKING_PT_BINS,
                            "displacement": TRACKING_DISPLACEMENT_BINS,
                        },
                        truth_min_hits=int(self.args.sweep_truth_min_hits),
                        min_theta=10.0,
                        max_theta=170.0,
                        gen_status=(0, 1),
                        rejected_seed_policy=self.rejected_seed_policy,
                    )
                )
                pt_count_tensor = torch.tensor(
                    np.stack([
                        np.stack(pair, axis=0)
                        for pair in comparison_rows["pt"]
                    ], axis=0),
                    dtype=torch.int64,
                    device=reduction_device,
                )
                displacement_count_tensor = torch.tensor(
                    np.stack([
                        np.stack(pair, axis=0)
                        for pair in comparison_rows["displacement"]
                    ], axis=0),
                    dtype=torch.int64,
                    device=reduction_device,
                )
                missing_tensor = torch.tensor(
                    [missing_info["pt"], missing_info["displacement"]],
                    dtype=torch.int64,
                    device=reduction_device,
                )
                pt_event_count = torch.tensor(
                    len(self._val_cache), dtype=torch.int64, device=reduction_device
                )
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(pt_count_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(displacement_count_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(missing_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(pt_event_count, op=dist.ReduceOp.SUM)

                if self.trainer.is_global_zero:
                    pt_image_paths = {}
                    displacement_image_paths = {}
                    pt_counts = pt_count_tensor.cpu().numpy()
                    displacement_counts = displacement_count_tensor.cpu().numpy()
                    for name in ("max_efficiency", "pareto_f1"):
                        pt_image_paths[name] = save_tracking_efficiency_pt_plot(
                            matching_comparison_plot_series(
                                comparison_order, pt_counts, name
                            ),
                            str(output_dir),
                            filename_stem=f"tracking_efficiency_vs_pt_{name}",
                            bins=TRACKING_PT_BINS,
                            min_x=0.1,
                            max_x=60.0,
                            min_theta=10.0,
                            max_theta=170.0,
                            gen_status=(0, 1),
                        )
                        displacement_image_paths[name] = (
                            save_tracking_efficiency_displacement_plot(
                                matching_comparison_plot_series(
                                    comparison_order,
                                    displacement_counts,
                                    name,
                                ),
                                str(output_dir),
                                filename_stem=(
                                    "tracking_efficiency_vs_displacement_" f"{name}"
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
                        "Saved double-majority and Hungarian tracking "
                        "efficiency versus pT and uniformly binned "
                        "displacement from "
                        f"{int(pt_event_count.item())} validation events; "
                        "missing pT/displacement metadata for "
                        f"{int(missing_tensor[0].item())}/"
                        f"{int(missing_tensor[1].item())} eligible truth tracks.",
                        flush=True,
                    )
                    if hasattr(self.logger, "log_image"):
                        wandb_media.update(tracking_efficiency_media(
                            pt_image_paths, displacement_image_paths
                        ))

            if (
                self.trainer.is_global_zero
                and wandb_media
                and hasattr(self.logger, "experiment")
            ):
                # Do not pass W&B's internal `step=` here.  WandbLogger defines
                # trainer/global_step as the synchronized metric step; an
                # explicit W&B step would create a second, competing counter.
                log_wandb_media(self.logger, wandb_media)

        if self._ema_restore_state is not None:
            self.load_state_dict(self._ema_restore_state, strict=False)
            self._ema_restore_state = None

    def _initialize_ema_if_needed(self):
        if self._ema is None and self._ema_decay > 0:
            self._ema = EMAShadow(self, self._ema_decay)
        if self._ema is not None and self._pending_ema_state is not None:
            self._ema.load_state_dict(self._pending_ema_state)
            self._pending_ema_state = None

    def on_fit_start(self):
        self._initialize_ema_if_needed()
        if self._ema is not None:
            # The standalone pre-training validation runs in inference mode.
            # EMA tensors created/restored there must be made mutable before
            # the first training batch updates them in place.
            self._ema.prepare_for_updates()

        if bool(getattr(self.args, "override_resume_lr", False)):
            requested_lr = float(self.args.start_lr)
            if requested_lr <= 0:
                raise ValueError("--start-lr must be greater than zero")

            overridden_groups = 0
            for optimizer in self.trainer.optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = requested_lr
                    group["initial_lr"] = requested_lr
                    overridden_groups += 1

            if overridden_groups == 0:
                raise RuntimeError(
                    "--override-resume-lr could not find a configured optimizer"
                )

            # Keep the live scheduler bookkeeping consistent with the forced
            # optimizer value. ReduceLROnPlateau will subsequently reduce from
            # START_LR while retaining its restored best/patience history.
            for config in self.trainer.lr_scheduler_configs:
                scheduler = config.scheduler
                if isinstance(getattr(scheduler, "base_lrs", None), list):
                    scheduler.base_lrs = [
                        requested_lr for _ in scheduler.base_lrs
                    ]
                if isinstance(getattr(scheduler, "_last_lr", None), list):
                    scheduler._last_lr = [
                        requested_lr for _ in scheduler._last_lr
                    ]

            if self.trainer.is_global_zero:
                print(
                    f"Forced {overridden_groups} optimizer parameter-group "
                    f"learning rate(s) to {requested_lr} after checkpoint restore.",
                    flush=True,
                )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log(
            "trainer/epoch",
            float(self.current_epoch),
            on_step=True,
            on_epoch=False,
            sync_dist=False,
            logger=False,
        )
        for optimizer_index, optimizer in enumerate(self.trainer.optimizers):
            optimizer_name = optimizer.__class__.__name__
            for group_index, group in enumerate(optimizer.param_groups):
                metric_name = f"trainer/lr-{optimizer_name}"
                if len(optimizer.param_groups) > 1:
                    metric_name += f"/group_{group_index}"
                if len(self.trainer.optimizers) > 1:
                    metric_name += f"/optimizer_{optimizer_index}"
                self.log(
                    metric_name,
                    float(group["lr"]),
                    on_step=True,
                    on_epoch=False,
                    sync_dist=False,
                    logger=False,
                )

    def on_save_checkpoint(self, checkpoint):
        if self._ema is not None:
            checkpoint["ema_state_dict"] = self._ema.state_dict()
        checkpoint["lr_schedule_unit"] = self.lr_schedule_unit
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

        checkpoint_schedule_unit = checkpoint.get("lr_schedule_unit")
        if (
            checkpoint_schedule_unit is not None
            and checkpoint_schedule_unit != self.lr_schedule_unit
        ):
            raise ValueError(
                "A full resume cannot change scheduler type: checkpoint uses "
                f"{checkpoint_schedule_unit!r}, requested {self.lr_schedule_unit!r}. "
                "Use --checkpoint-mode weights to start a new optimizer and scheduler."
            )

    def configure_optimizers(self):
        options = {}
        for key, value in getattr(self.args, "optimizer_option", []):
            options[key] = ast.literal_eval(value)
        options.setdefault("weight_decay", float(getattr(self.args, "weight_decay", 1e-4)))
        options.setdefault(
            "fused", str(getattr(self.args, "precision", "32-true")) == "32-true"
        )
        start_lr = float(self.args.start_lr)
        if start_lr <= 0:
            raise ValueError("--start-lr must be greater than zero")
        name = str(self.args.optimizer).lower()
        if name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=start_lr, **options)
        elif name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=start_lr, **options)
        elif name == "radam":
            optimizer = torch.optim.RAdam(self.parameters(), lr=start_lr, **options)
        else:
            raise ValueError("Lightning GATr supports optimizer adamW, adam, or radam")

        scheduler_name = str(
            getattr(self.args, "lr_scheduler", "reduceplateau")
        ).lower()
        if scheduler_name == "none":
            return optimizer
        if self._steps_per_epoch <= 0:
            estimated = self.trainer.estimated_stepping_batches
            if not math.isfinite(float(estimated)):
                raise RuntimeError(
                    f"{scheduler_name} requires --steps-per-epoch when the "
                    "training loader has no finite length"
                )
            self._steps_per_epoch = int(
                estimated // max(int(self.args.num_epochs), 1)
            )
        if self._steps_per_epoch <= 0:
            raise RuntimeError(
                f"{scheduler_name} requires a finite steps-per-epoch"
            )

        if scheduler_name == "reduceplateau":
            self._warmup_steps = int(
                float(getattr(self.args, "warmup_epochs", 2.0))
                * self._steps_per_epoch
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(getattr(self.args, "plateau_factor", 0.5)),
                patience=int(getattr(self.args, "plateau_patience", 3)),
                threshold=float(getattr(self.args, "plateau_threshold", 1e-4)),
                threshold_mode="rel",
                min_lr=self._min_lr,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "validation/loss",
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": True,
                },
            }
        if scheduler_name != "flat+decay":
            raise ValueError(
                "Lightning GATr supports lr_scheduler none, flat+decay, "
                "or reduceplateau"
            )

        total_steps = max(int(self.args.num_epochs), 1) * self._steps_per_epoch
        warmup_steps = int(
            float(getattr(self.args, "warmup_epochs", 2.0))
            * self._steps_per_epoch
        )
        min_ratio = float(getattr(self.args, "min_lr", 1e-6)) / start_lr

        def schedule(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(
                total_steps - warmup_steps, 1
            )
            return max(
                min_ratio,
                0.5 * (1.0 + math.cos(math.pi * progress)),
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
