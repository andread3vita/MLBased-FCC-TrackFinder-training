"""GGTF's track finder with the conformal algebra in place of the projective one.

A line-for-line counterpart of `Gatr_withModifications.py`. The data pipeline, the
object-condensation loss, the batch norm, the attention masking, the optimiser and the
schedule are all theirs and unchanged; only the algebra moves. That is the entire point
of the file, so please keep it that way -- every extra difference is a confound that has
to be argued away later.

Why this exists. Our own five-arm comparison put a projective backbone using GGTF's hit
encoding ahead of every conformal arm we have, but the two projective arms differed from
each other by far more than the algebras differed (validation loss 0.769 against 0.429,
same algebra, same capacity, same schedule, only the hit encoding changed). So that
comparison measures the encoding, not the algebra, and cannot answer whether the
conformal algebra is the better one. Holding their encoding fixed and swapping only the
algebra is the comparison that can. See
`paper_adjustment_candidates/curler_question.md`.

What necessarily differs, and why it is not a confound:

*Blade count.* Conformal Cl(4,1) has 32 blades against the projective 16, so the two
readout heads take 32 inputs rather than 16. Unavoidable.

*Width.* Equal `hidden_mv_channels` would hand the conformal arm far more parameters,
because it has both twice the blades and 40 equivariant linear maps against 16. The
width defaults to CIRCE's reference (16); `--capacity-matched` selects the width
calibrated to their parameter count instead; see `HIDDEN_MV_MATCHED`.

*Attention.* Theirs adds the hand-crafted distance features of `_build_dist_basis`,
which exist because the projective inner product between two points is constant in
their coordinates (de Haan et al. Prop. 3) and attention would otherwise be blind to
position. In a non-degenerate algebra that problem does not arise: the invariant inner
product between two conformal null vectors already *is* the squared Euclidean distance,
which is the stated reason that paper prefers this algebra (Sec. 4.3). So the default
here is the weighted invariant inner product and no distance features. Forcing their
features onto an algebra that does not need them would be the wrong comparison, but
`dist_basis=True` is available to check that claim rather than assume it.

*Translation embedding.* `embed_translation` had to be written for the conformal algebra
(`cgatr/interface/translation.py`). It is the same rotor construction as theirs with the
point at infinity generating the translation instead of the degenerate `e0`, verified by
sandwich product against `embed_point(x + t)`. Note that CGA could encode a drift circle
directly, which is strictly more expressive than a point plus an offset; we deliberately
do not, because using their encoding is what makes this an algebra comparison.
"""

import torch
import torch.nn as nn
import lightning as L
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.logger.logger_wandb import log_losses_wandb_tracking
from src.layers.inference_oc_tracks import (
    evaluate_efficiency_tracks,
    store_at_batch_end,
    store_at_batch_end_hits
)
from src.layers.losses import object_condensation_loss_tracking
from src.layers.losses_circe import circe_condensation_loss
from src.layers.batch_operations import obtain_batch_numbers
from src.layers.validation_metrics import compute_batch_metrics_greedy

from src.cgatr.nets.cgatr import CGATr
from src.cgatr.layers.attention.config import SelfAttentionConfig
from src.cgatr.layers.mlp.config import MLPConfig
from src.cgatr.interface.point import embed_point
from src.cgatr.interface.scalar import embed_scalar
from src.cgatr.interface.translation import embed_translation
from src.cgatr.primitives.invariants import compute_inner_product_mask
from src.cgatr.primitives.linear import _compute_se3_equi_linear_basis
from src.cgatr.primitives.attention import _build_dist_basis
from src.cgatr.primitives.dual import _DualCache

NUM_BLADES = 32

# Calibrated against their projective model, which is mv=16 at 10 blocks and 64 scalar
# channels and comes to 924,488 parameters. Width 9 lands at 937,657, i.e. 1.4% above
# theirs, and is the closest available match -- 8 is 8.8% below and 10 is 12.5% above,
# the steps being coarse because each conformal channel carries 32 blades and 40
# equivariant maps. Recompute with `python -m src.models.report_algebra_params` if their
# widths ever change; it prints both arms side by side and picks the match.
HIDDEN_MV_REFERENCE = 16  # CIRCE's production width (what our results use)
HIDDEN_MV_MATCHED = 9     # matches Gatr_withModifications' 924,488 params within 1.4%


class ExampleWrapper(L.LightningModule):
    def __init__(
        self,
        args,
        dist_basis: bool = False,
    ):
        super().__init__()
        blocks = 10
        # Hardcoded, not read from `args`, because `Gatr_withModifications` hardcodes its
        # own 16 and the two arms have to be equally immune to their surroundings. Reading
        # args here would mean a stray `--hidden_mv_channels` moved the conformal width
        # while leaving the projective one at 16, quietly turning an algebra comparison
        # into a capacity comparison. Three standalone trainers in this fork
        # (train_cgatr_parquet, train_cgatr_lightning_v35, eval_sweep_v33) do define that
        # flag with default 16, so this is a live hazard rather than a hypothetical --
        # though the A/B runs through `src.train_lightning`, whose parser has no width
        # argument at all.
        matched = bool(getattr(args, "capacity_matched", False))
        hidden_mv_channels = HIDDEN_MV_MATCHED if matched else HIDDEN_MV_REFERENCE
        hidden_s_channels = 64
        print(f"[cgatr] conformal arm at hidden_mv_channels={hidden_mv_channels}, "
              f"blocks={blocks}, hidden_s_channels={hidden_s_channels} "
              + ("(capacity-matched to GGTF's projective 924,488 within 1.4%)"
                 if matched else "(CIRCE reference width)"))
        self.input_dim = 3
        self.output_dim = 4
        self.args = args
        self.dist_basis = dist_basis
        self.basis_gp = None
        self.basis_outer = None
        self.pin_basis = None
        self.basis_ip_weights = None
        self.basis_q = None
        self.basis_k = None
        self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(self.input_dim, momentum=0.1)

        self.load_basis()

        if self.dist_basis:
            attention = SelfAttentionConfig()
        else:
            # Attention scores are the invariant inner product <x~ y>_0 = sum_i w_i x_i
            # y_i. Sixteen of the conformal weights are -1, so unlike the projective
            # case this cannot be shortened to a plain dot product over a subset of
            # blades; the indices and weights both have to be handed over.
            ip_idx = torch.nonzero(self.basis_ip_weights, as_tuple=True)[0].tolist()
            ip_weights = self.basis_ip_weights[ip_idx].tolist()
            attention = SelfAttentionConfig(ip_idx=ip_idx, ip_weights=ip_weights)

        self.cgatr = CGATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=None,
            out_s_channels=None,
            hidden_s_channels=hidden_s_channels,
            num_blocks=blocks,
            attention=attention,
            mlp=MLPConfig(),
            basis_gp=self.basis_gp,
            basis_ip_weights=self.basis_ip_weights,
            basis_outer=self.basis_outer,
            basis_pin=self.pin_basis,
            basis_q=self.basis_q,
            basis_k=self.basis_k,
        )

        self.clustering = nn.Linear(NUM_BLADES, self.output_dim - 1, bias=False)
        self.beta = nn.Linear(NUM_BLADES, 1)
        self.vector_like_data = True
        self._val_metrics = []

    def load_basis(self):
        # Their version pins these to "cuda" at load time. Resolving the device instead
        # keeps the parameter-count check and any CPU smoke test runnable; on a GPU run
        # it resolves to the same place.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        filename = "cga_utils/cga_geometric_product.pt"
        sparse_basis = torch.load(filename, weights_only=False).to(torch.float32)
        basis = sparse_basis.to_dense()
        self.basis_gp = basis.to(device=device)
        filename = "cga_utils/cga_outer_product.pt"
        sparse_basis_outer = torch.load(filename, weights_only=False).to(torch.float32)
        sparse_basis_outer = sparse_basis_outer.to_dense()
        self.basis_outer = sparse_basis_outer.to(device=device)

        metadata = torch.load("cga_utils/cga_metadata.pt", weights_only=False)
        _DualCache.init_from_metadata(metadata, device=device)

        # The equivariant linear basis is the null space of the Lie-algebra constraint,
        # and which null direction generates translations is the whole question this
        # arm is here to answer correctly. inf = e- - e+ is fixed by the point
        # embedding P = o + p + |p|^2 inf/2 in interface/point.py. The other null
        # direction o = (e+ + e-)/2 generates transversions, not translations, and
        # produces a basis sharing only 12 of its 40 maps -- the bug recorded in
        # paper_adjustment_candidates/drift_embedding_null_vector.md.
        translation_vec = torch.zeros(NUM_BLADES)
        translation_vec[4] = -1.0  # e+
        translation_vec[5] = 1.0   # e-
        self.pin_basis = _compute_se3_equi_linear_basis(
            self.basis_gp, device=device, dtype=basis.dtype,
            spatial_idx=(1, 2, 3),
            translation_idx=None,
            translation_vec=translation_vec,
            label="CGA",
        )

        # reversal=None: the projective mask needs the reversal signs to spot its
        # degenerate blades, the conformal metric is non-degenerate and does not.
        self.basis_ip_weights = compute_inner_product_mask(
            self.basis_gp, device=device, reversal=None,
        )

        if self.dist_basis:
            self.basis_q, self.basis_k = _build_dist_basis(
                device=device, dtype=basis.dtype
            )

    def forward(self, g, input):

        pos_hits_xyz = input[:, 0:3]
        hit_type = input[:, 3].view(-1, 1)
        vector = input[:, 4:]

        inputs = self.ScaledGooeyBatchNorm2_1(pos_hits_xyz)
        velocities = embed_translation(vector)
        embedded_inputs = embed_point(inputs) + embed_scalar(hit_type) + velocities

        embedded_inputs = embedded_inputs.unsqueeze(-2)
        scalars = torch.zeros((inputs.shape[0], 1), device=inputs.device)
        mask = self.build_attention_mask(g)

        embedded_outputs, _ = self.cgatr(
            embedded_inputs, scalars=scalars, attention_mask=mask
        )
        output = embedded_outputs[:, 0, :]
        x_cluster_coord = self.clustering(output)
        beta = self.beta(output)
        x = torch.cat((x_cluster_coord, beta), dim=1)

        return x

    def build_attention_mask(self, g):
        """Per-event hit counts, which is what the conformal attention wants.

        Their projective version returns `BlockDiagonalMask.from_seqlens(...)` here.
        The conformal attention takes the sequence lengths themselves and builds that
        same xformers mask internally, so this hands over the list and the mask is
        constructed one layer down. Identical attention pattern, and it is the form
        that avoids materialising the dense score matrix: a 32-blade multivector times
        the channel count puts the per-head feature dimension well above the 256-dim
        cap of the Flash and memory-efficient SDPA kernels, so a dense mask would fall
        back to the MATH backend and allocate the full M x M matrix including the
        cross-event entries it then zeroes.
        """
        batch_numbers = obtain_batch_numbers(g)
        return torch.bincount(batch_numbers.long()).tolist()

    def training_step(self, batch, batch_idx):
        y = batch[1]
        batch_g = batch[0]

        pos_hits_xyz = batch_g.ndata["pos_hits_xyz"]
        hit_type = batch_g.ndata["hit_type"].view(-1, 1)
        vector = batch_g.ndata["vector"]
        input_ = torch.cat((pos_hits_xyz, hit_type, vector), dim=1)

        model_output = self(batch_g, input_)

        # CIRCE trains with its own objective by default; set
        # args.loss_backend = "ggtf" for algebra-only A/B runs where both
        # arms must share the exact same loss (train_algebra_ab does this).
        if getattr(self.args, "loss_backend", "circe") == "circe":
            (loss, losses) = circe_condensation_loss(
                batch_g, model_output, y, self.args
            )
        else:
            (loss, losses) = object_condensation_loss_tracking(
                batch_g,
                model_output,
                y,
                clust_loss_only=True,
                add_energy_loss=False,
                calc_e_frac_loss=False,
                q_min=self.args.qmin,
                frac_clustering_loss=self.args.frac_cluster_loss,
                attr_weight=self.args.L_attractive_weight,
                repul_weight=self.args.L_repulsive_weight,
                fill_loss_weight=self.args.fill_loss_weight,
                use_average_cc_pos=self.args.use_average_cc_pos,
                loss_type= self.args.loss_type,
                tracking=True,
            )

        if torch.isnan(loss):
            print(f"Batch {batch_idx} returns NaN, skip.")
            return None

        if self.trainer.is_global_zero:
            log_losses_wandb_tracking(True, batch_idx, 0, losses, loss)

        return loss

    def validation_step(self, batch, batch_idx):
        self.validation_step_outputs = []
        y = batch[1]

        batch_g = batch[0]

        pos_hits_xyz = batch_g.ndata["pos_hits_xyz"]
        hit_type = batch_g.ndata["hit_type"].view(-1, 1)
        vector = batch_g.ndata["vector"]
        input_ = torch.cat((pos_hits_xyz, hit_type, vector), dim=1)

        model_output = self(batch_g, input_)
        dic = {}
        batch_g.ndata["model_output"] = model_output
        dic["graph"] = batch_g
        dic["part_true"] = y

        # CIRCE trains with its own objective by default; set
        # args.loss_backend = "ggtf" for algebra-only A/B runs where both
        # arms must share the exact same loss (train_algebra_ab does this).
        if getattr(self.args, "loss_backend", "circe") == "circe":
            (loss, losses) = circe_condensation_loss(
                batch_g, model_output, y, self.args
            )
        else:
            (loss, losses) = object_condensation_loss_tracking(
                batch_g,
                model_output,
                y,
                clust_loss_only=True,
                add_energy_loss=False,
                calc_e_frac_loss=False,
                q_min=self.args.qmin,
                frac_clustering_loss=self.args.frac_cluster_loss,
                attr_weight=self.args.L_attractive_weight,
                repul_weight=self.args.L_repulsive_weight,
                fill_loss_weight=self.args.fill_loss_weight,
                use_average_cc_pos=self.args.use_average_cc_pos,
                loss_type= self.args.loss_type,
                tracking=True,
            )
        if self.trainer.is_global_zero:
            log_losses_wandb_tracking(True, batch_idx, 0, losses, loss, val=True)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

        metric_batches = int(getattr(self.args, "metric_val_batches", 40))
        if metric_batches <= 0 or batch_idx < metric_batches:
            self._val_metrics.append(compute_batch_metrics_greedy(
                model_output[:, :-1],
                model_output[:, -1:],
                batch_g.ndata["particle_number"],
                batch_g.ndata["isSecondary"].view(-1),
                batch_g.batch_num_nodes().tolist(),
                tbeta=float(getattr(self.args, "tbeta", 0.1)),
                td=float(getattr(self.args, "td", 0.2)),
            ))

        if self.trainer.is_global_zero and self.args.predict:
            df_batch, df_hits = evaluate_efficiency_tracks(
                batch_g,
                model_output,
                y,
                0,
                batch_idx,
                0,
                path_save=self.args.model_prefix + "showers_df_evaluation",
                store=True,
                predict=False,
                tau=self.args.tau

            )
            if self.args.predict:
                if len(df_batch) > 0:
                    self.df_showers.append(df_batch)

                if len(df_hits) > 0:
                    self.df_showers_hits.append(df_hits)

    def on_validation_epoch_start(self):
        self.make_mom_zero()
        self._val_metrics = []
        self.df_showers = []
        self.df_showers_hits = []
        self.df_showers_pandora = []
        self.df_showes_db = []

    def make_mom_zero(self):
        if self.current_epoch > 2 or self.args.predict:
            self.ScaledGooeyBatchNorm2_1.momentum = 0

    def on_validation_epoch_end(self):
        loose = (
            float(np.mean([m["match_rate"] for m in self._val_metrics]))
            if self._val_metrics else 0.0
        )
        strict50 = (
            float(np.mean([
                m["match_rate_strict50"] for m in self._val_metrics
            ]))
            if self._val_metrics else 0.0
        )
        log_kwargs = dict(on_epoch=True, sync_dist=True, batch_size=1)
        self.log("val/match_rate", loose, **log_kwargs)
        self.log("val/match_rate_strict50", strict50, **log_kwargs)

        if self.args.predict:
            store_at_batch_end(
                self.args.model_prefix + "showers_df_evaluation",
                self.df_showers,
                0,
                0,
                0,
                predict=True,
            )

            store_at_batch_end_hits(
                self.args.model_prefix + "showers_df_evaluation",
                self.df_showers_hits,
                0,
                0,
                0,
                predict=True,
            )

    def configure_optimizers(self):
        # CIRCE's reference recipe (the settings its results are reported
        # with): AdamW + weight decay, 2 warm-up epochs, flat, then a
        # half-cosine anneal to min_lr over the last anneal_epochs epochs.
        # Select with --recipe circe (default for this model); --recipe ggtf
        # keeps the original Adam + ReduceLROnPlateau below.
        if getattr(self.args, "recipe", "circe") == "circe":
            import math

            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.args.start_lr,
                weight_decay=float(getattr(self.args, "weight_decay", 1e-4)),
            )
            total = int(getattr(self.args, "num_epochs", 30))
            warmup = int(getattr(self.args, "warmup_epochs", 2))
            anneal = min(int(getattr(self.args, "anneal_epochs", 6)),
                         max(total - warmup, 1))
            start = total - anneal
            min_frac = float(getattr(self.args, "min_lr", 1e-5)) / float(
                self.args.start_lr)

            def lr_lambda(epoch):
                if epoch < warmup:
                    return float(epoch + 1) / max(warmup, 1)
                if epoch >= start:
                    prog = min(max(float(epoch - start) / anneal, 0.0), 1.0)
                    return min_frac + 0.5 * (1.0 - min_frac) * (
                        1.0 + math.cos(math.pi * prog))
                return 1.0

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.start_lr)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            threshold=1e-3,
            verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
