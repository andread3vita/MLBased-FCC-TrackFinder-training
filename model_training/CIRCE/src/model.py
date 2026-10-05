"""C-GATr FCC track-finding model — simplified, M1-M4 baked in.

All ablation flags removed; these improvements are now permanent:
  M1: grade-wise equivariant LayerNorm (normalization.py)
  M2: drift-radius fixed-scale normalization (dc[:,7]/5.0)
  M3: isotropic position normalization (pos/1000.0)
  M4: faithful OC loss (Kieseler 2002.03605 hinge repulsive + arctanh^2)

New symmetry-preserving runs return clustering coordinates and beta through
GATr's scalar output representation. The historical unconstrained blade-mixing
readout remains only for loading old checkpoints.
"""

from __future__ import annotations

import os
import sys
import copy
import math
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch_scatter import scatter_max, scatter_add, scatter_mean

from src.cgatr.layers.linear import EquiLinear
from src.cgatr.nets.cgatr import CGATr
from src.cgatr.layers.attention.config import SelfAttentionConfig
from src.cgatr.layers.mlp.config import MLPConfig
from src.cgatr.interface.point import embed_point
from src.cgatr.interface.scalar import embed_scalar
from src.cgatr.interface.circle import embed_circle_ipns
from src.cgatr.interface.line import embed_line
from src.cgatr.interface.sphere import embed_sphere
from src.cgatr.interface.plane import embed_plane
from src.cgatr.primitives.linear import (
    _compute_e3_equi_linear_basis,
    _compute_se3_equi_linear_basis,
)
from src.cgatr.primitives.attention import _build_dist_basis, block_diagonal_bool_mask
from src.cgatr.primitives.invariants import compute_inner_product_mask
from src.cgatr.primitives.dual import _DualCache
from src.dataset.parquet_dataset import IDEAParquetDataset, collate_idea_events


class CGATrParquetModel(nn.Module):
    """C-GATr model with M1-M5 baked in. Train from scratch (--init_weights none).

    VTX hits  -> CGA null point  (grade-1 vector,  32-dim MV)
    DC  hits  -> IPNS circle     (grade-2 bivector, 32-dim MV)
    Both in ONE multivector channel.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        # M3: isotropic norm — single scale, preserves E(3) rotation equivariance.
        self.pos_scale = 1000.0

        # Hit time enters as a scalar channel rather than through embed_scalar:
        # the grade-0 blade already carries hit_type, and summing the two there
        # would make them indistinguishable.
        self.use_time = bool(getattr(args, "use_time", False))
        self.algebra = getattr(args, "algebra", "conformal")
        if self.algebra not in ("conformal", "projective"):
            raise ValueError(f"unknown algebra {self.algebra!r}")
        self.projective = self.algebra == "projective"
        self.pga_encoding = getattr(args, "pga_hit_encoding", "line")
        if self.pga_encoding not in ("line", "ggtf", "ggtf_wire", "point_line"):
            raise ValueError(f"unknown pga_hit_encoding {self.pga_encoding!r}")
        # Only GGTF's encoding needs the left-to-right drift vector, so only it
        # asks the dataset for the extra feature columns.
        self.needs_drift_dir = (self.projective
                                and self.pga_encoding in ("ggtf", "ggtf_wire"))
        self.physical_drift_geometry = bool(
            getattr(args, "physical_drift_geometry", False))
        self.separate_hit_metadata = bool(
            getattr(args, "separate_hit_metadata", False))
        # A clean hit-type-location ablation: unlike
        # --separate_hit_metadata, this does not add a redundant drift-radius
        # scalar beside the geometric sphere/circle radius.
        self.separate_hit_type = bool(
            self.separate_hit_metadata
            or getattr(args, "separate_hit_type", False)
        )
        self.layernorm_epsilon_mode = getattr(
            args, "layernorm_epsilon_mode", "clamp"
        )
        if self.layernorm_epsilon_mode not in ("clamp", "add"):
            raise ValueError(
                "unknown layernorm_epsilon_mode "
                f"{self.layernorm_epsilon_mode!r}"
            )
        self.equi_init = getattr(args, "equi_init", "default")
        if self.equi_init not in ("default", "identity_algebra"):
            raise ValueError(f"unknown equi_init {self.equi_init!r}")
        self.equivariance_group = getattr(args, "equivariance_group", "se3").lower()
        if self.equivariance_group not in ("e3", "se3"):
            raise ValueError(
                f"unknown equivariance_group {self.equivariance_group!r}")
        self.invariant_output_head = bool(
            getattr(args, "invariant_output_head", False)
        )

        # Two-channel drift embedding. Today a drift hit is a grade-2 circle
        # while a vertex hit is a grade-1 point, so the conformal inner product
        # between them is not the point-to-sphere distance that motivates the
        # algebra in the first place -- it is a circle-point product of mixed
        # grade. Giving each drift hit a sphere channel alongside its circle puts
        # the two hit types back on a shared grade, so the distance the network
        # can read between a vertex hit and a drift hit is the clean one, while
        # the circle channel still carries the full wire constraint.
        # The sphere and plane embeddings offset the centre along e+ + e-, which
        # is 2o, the origin, where the definition calls for inf = e- - e+. It is
        # the same o-versus-inf confusion the equivariant linear basis had before
        # --no_legacy_equivariance, in a second and independent place. Off by
        # default because every conformal checkpoint so far trained on those
        # inputs; see src/cgatr/interface/sphere.py for what it costs.
        self.fix_cga_null = getattr(args, "fix_cga_null", False)

        # The wire direction was built with the azimuth in the wrong slots,
        # (sin s cos a, sin s sin a, cos s), which tilts the wire radially. A
        # stereo wire tilts azimuthally: the detector convention that produced
        # left/right in the parquet is (sin s sin a, -sin s cos a, cos s). The
        # wrong vector is a median 10.8 degrees off, up to 20.2, and is not
        # perpendicular to the drift direction the data carries. Off by default
        # because every conformal checkpoint so far trained on the tilted plane.
        self.fix_wire_dir = getattr(args, "fix_wire_dir", False)

        self.two_channel_dc = getattr(args, "two_channel_dc", False)
        if self.two_channel_dc and self.projective:
            raise ValueError(
                "--two_channel_dc is a conformal construction: it needs the "
                "grade-1 sphere, which the projective algebra has no room for")
        self.cga_encoding = getattr(args, "cga_hit_encoding", "circle")
        if self.cga_encoding not in (
            "circle", "sphere_circle", "sphere_plane", "point_line"
        ):
            raise ValueError(f"unknown cga_hit_encoding {self.cga_encoding!r}")
        if self.two_channel_dc:
            if self.cga_encoding != "circle":
                raise ValueError(
                    "--two_channel_dc is the legacy spelling of "
                    "--cga_hit_encoding sphere_circle; do not pass both")
            self.cga_encoding = "sphere_circle"
        if self.projective and self.cga_encoding != "circle":
            raise ValueError("--cga_hit_encoding applies only to conformal models")
        if self.cga_encoding == "point_line":
            if not self.fix_wire_dir:
                raise ValueError(
                    "CGA point_line requires --fix_wire_dir"
                )
            if not self.separate_hit_metadata:
                raise ValueError(
                    "CGA point_line requires --separate_hit_metadata so drift "
                    "radius and hit type are not discarded"
                )
        prefix = "pga" if self.projective else "cga"

        gp_sparse = torch.load(f"cga_utils/{prefix}_geometric_product.pt", weights_only=False)
        self.register_buffer("basis_gp", gp_sparse.to_dense().to(dtype=torch.float32))

        op_sparse = torch.load(f"cga_utils/{prefix}_outer_product.pt", weights_only=False)
        self.register_buffer("basis_outer", op_sparse.to_dense().to(dtype=torch.float32))

        metadata = torch.load(f"cga_utils/{prefix}_metadata.pt", weights_only=False)
        _DualCache.init_from_metadata(metadata, device=torch.device("cpu"))
        self.num_blades = int(metadata["num_blades"])
        self._blade_names = list(metadata["blade_names"])
        # A plain attribute, not a buffer: it is only used by the equivariance
        # test, and adding a buffer would put a new key in the state dict and
        # break loading of every existing checkpoint.
        self._reversal = metadata["reversal_signs"].clone()

        # Reproduces the pre-2026-07-31 conformal architecture, whose backbone is
        # rotation- but not translation-equivariant. Kept only so the ablation
        # can measure what correcting it buys; see
        # paper_adjustment_candidates/equivariance_claim.md.
        self.legacy_equi = bool(getattr(args, "legacy_equivariance", False))
        if self.legacy_equi and self.equivariance_group != "se3":
            raise ValueError(
                "--legacy_equivariance reproduces the old 40-map SE(3) "
                "backbone and cannot be combined with --equivariance_group e3")

        # Equivariant linear basis from de Haan et al. 2311.04744 Sec. 3.3.
        # The paper's full E(3) construction has 20 CGA maps; removing the
        # mirror constraint gives 40 SE(3) maps and allows chirality in the
        # fixed solenoidal field. Both are computed from the same verified
        # Lie-algebra null space.
        #
        # Translations are generated against the degenerate e0 in the projective
        # algebra and against the point at infinity inf = e- - e+ in the
        # conformal one. inf is fixed by the point embedding P = o + p +
        # |p|^2 inf/2 in interface/point.py. The other null direction, o =
        # (e+ + e-)/2, generates transversions instead, and yields an equally
        # large basis sharing only 12 of its 40 maps -- which is what the legacy
        # path used.
        translation_vec = None
        if not self.projective:
            translation_vec = torch.zeros(int(metadata["num_blades"]))
            translation_vec[4] = 1.0 if self.legacy_equi else -1.0  # e+
            translation_vec[5] = 1.0                                # e-
        spatial_idx = (2, 3, 4) if self.projective else (1, 2, 3)
        translation_idx = 1 if self.projective else None
        basis_kwargs = dict(
            gp=self.basis_gp,
            device=torch.device("cpu"),
            dtype=torch.float32,
            spatial_idx=spatial_idx,
            translation_idx=translation_idx,
            translation_vec=translation_vec,
            label="PGA" if self.projective else "CGA",
        )
        if self.equivariance_group == "e3":
            pin_basis = _compute_e3_equi_linear_basis(
                grade_involution=metadata["grade_involution_signs"],
                mirror_vector_idx=spatial_idx[0],
                expected_count=9 if self.projective else 20,
                **basis_kwargs,
            )
        else:
            pin_basis = _compute_se3_equi_linear_basis(**basis_kwargs)
        basis_ip_weights = compute_inner_product_mask(
            self.basis_gp, device=torch.device("cpu"),
            reversal=metadata["reversal_signs"] if self.projective else None,
        )
        # Attention scores are the invariant inner product <x~ y>_0 = sum_i
        # w_i x_i y_i, so the blades with zero weight are dropped and the rest
        # are weighted. In the projective algebra the weights are all +1 and the
        # zeros are exactly the e0 blades, which recovers GATr's "plain dot
        # product on 8 of the 16 dimensions". In the conformal algebra sixteen
        # weights are -1, so the same unweighted shortcut is not invariant.
        ip_idx = torch.nonzero(basis_ip_weights, as_tuple=True)[0].tolist()
        ip_weights = basis_ip_weights[ip_idx].tolist()

        if self.projective:
            self.register_buffer("point_matrix", metadata["point_matrix"])
            self.register_buffer("line_matrix", metadata["line_matrix"])
            # GATr's translation embedding, T(t) = 1 - e0 (t . e) / 2
            # (gatr/interface/translation.py). Built by blade name so it does
            # not depend on our ordering, which swaps e12 and e03 relative to
            # the reference.
            blade = {str(n): i for i, n in enumerate(metadata["blade_names"])}
            trans_matrix = torch.zeros(int(metadata["num_blades"]), 4)
            trans_matrix[blade["1"], 3] = 1.0
            for axis, name in enumerate(("e01", "e02", "e03")):
                trans_matrix[blade[name], axis] = -0.5
            self.register_buffer("translation_matrix", trans_matrix)
            basis_q = basis_k = None
            # Distance-aware attention is not optional here. The projective
            # inner product between two points is provably constant in their
            # coordinates (de Haan et al. Prop. 3), so without these features
            # attention is blind to position and the arm degenerates into the
            # P-GATr variant their paper reports as the weak one. With them,
            # plus the join, this is iP-GATr -- the architecture GGTF uses.
            names = [str(n) for n in self._blade_names]
            # GATr drops the trivector from the inner product once the distance
            # features are switched on -- the reference implementation attends
            # over `_INNER_PRODUCT_WO_TRI_IDX`, seven blades rather than eight,
            # because the trivector is what the distance features are built
            # from and would otherwise enter the logit twice.
            tri_blade = names.index("e123")
            ip_wo_tri = [(i, w) for i, w in zip(ip_idx, ip_weights)
                         if i != tri_blade]
            attention = SelfAttentionConfig(
                grade1_idx=[1, 2, 3, 4],
                ip_idx=[i for i, _ in ip_wo_tri],
                ip_weights=[w for _, w in ip_wo_tri],
                num_blades=self.num_blades,
                # Homogeneous weight first, then the components GATr writes as
                # q_{\1}, q_{\2}, q_{\3}.
                pga_dist_idx=[names.index(b)
                              for b in ("e123", "e023", "e013", "e012")],
            )
            mlp = MLPConfig(use_join=True, pseudoscalar_idx=self.num_blades - 1)
            # The projective algebra cannot hold a circle, so the drift radius
            # has to go somewhere. Under 'line' it becomes a scalar channel;
            # under 'ggtf' it stays geometric, as the length of the translation
            # between the two tangency points, and the scalar channel is then
            # only needed if hit time is switched on -- which is also why GGTF
            # itself runs with in_s_channels=None.
            include_drift_scalar = (
                self.separate_hit_metadata
                or self.pga_encoding in ("line", "point_line")
            )
            in_s_channels = (
                int(include_drift_scalar)
                + int(self.separate_hit_type)
                + int(self.use_time)
            ) or None
        elif self.legacy_equi:
            basis_q, basis_k = _build_dist_basis(
                device=torch.device("cpu"), dtype=torch.float32
            )
            attention = SelfAttentionConfig()
            mlp = MLPConfig()
            in_s_channels = (
                int(self.separate_hit_metadata)
                + int(self.separate_hit_type)
                + int(self.use_time)
            ) or None
        else:
            # C-GATr proper: no hand-crafted distance features. In a
            # non-degenerate algebra the invariant inner product between null
            # vectors already is the Euclidean distance, which is de Haan et
            # al.'s stated reason (Sec. 4.3) for preferring this algebra.
            basis_q = basis_k = None
            attention = SelfAttentionConfig(ip_idx=ip_idx, ip_weights=ip_weights)
            mlp = MLPConfig()
            in_s_channels = (
                int(self.separate_hit_metadata)
                + int(self.separate_hit_type)
                + int(self.use_time)
            ) or None

        self.cgatr = CGATr(
            in_mv_channels=2 if (
                self.cga_encoding in (
                    "sphere_circle", "sphere_plane", "point_line"
                )
                or self.pga_encoding in ("ggtf_wire", "point_line")
            ) else 1,
            out_mv_channels=1,
            hidden_mv_channels=args.hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=(
                args.embed_dim + 1 if self.invariant_output_head else None
            ),
            hidden_s_channels=args.hidden_s_channels,
            num_blocks=args.num_blocks,
            attention=attention,
            mlp=mlp,
            basis_gp=self.basis_gp,
            basis_ip_weights=basis_ip_weights,
            basis_outer=self.basis_outer,
            basis_pin=pin_basis,
            basis_q=basis_q,
            basis_k=basis_k,
            checkpoint_blocks=getattr(args, "grad_checkpoint", False),
            norm_epsilon_mode=self.layernorm_epsilon_mode,
        )

        if self.equi_init == "identity_algebra":
            # Applied after construction so only the layers that took the generic
            # scheme are touched; the bilinears' almost_unit_scalar layers are
            # tuned for their own job and are left as they are.
            n = 0
            for mod in self.cgatr.modules():
                if isinstance(mod, EquiLinear) and mod._init_kind == "default":
                    mod.reset_parameters("identity_algebra")
                    n += 1
            print(f"[cgatr] identity-on-algebra initialization applied to "
                  f"{n} equivariant linear layers", flush=True)

        self.embed_dim = args.embed_dim
        if self.invariant_output_head:
            self.clustering = None
            self.beta = None
        else:
            # Backward-compatible path for existing checkpoints. These heads
            # mix blades without an equivariance constraint and must not be used
            # for new symmetry-preserving runs.
            self.clustering = nn.Linear(
                self.num_blades, self.embed_dim, bias=False
            )
            self.beta = nn.Linear(self.num_blades, 1)

    @property
    def embedding_dim(self) -> int:
        """Alias for downstream evaluation scripts expecting `model.embedding_dim`."""
        return self.embed_dim

    def split_output(self, output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split model output into condensation coordinates and beta logits.

        Returns:
            coords: Tensor of shape (N, embed_dim) in Euclidean condensation space.
            beta_logits: Tensor of shape (N,) raw condensation logits.
        """
        coords = output[:, :self.embed_dim]
        beta_logits = output[:, self.embed_dim]
        return coords, beta_logits

    def forward(self, features, seq_lens):
        mv, scalars = self.embed(features)
        out_mv, out_scalars = self._backbone_outputs(mv, scalars, seq_lens)
        if self.invariant_output_head:
            if out_scalars is None:
                raise RuntimeError(
                    "invariant output head requested but GATr returned no scalars"
                )
            return out_scalars
        out = out_mv[:, 0, :]
        return torch.cat([self.clustering(out), self.beta(out)], dim=1)

    def embed(self, features):
        """Hits to multivectors and scalar channels. Split out of `forward` so
        the equivariance test can check the embedding and the stack separately.
        """
        pos = features[:, :3]
        hit_type = features[:, 3:4]
        # M3: isotropic norm
        pos_normed = pos / self.pos_scale

        is_vtx = (hit_type.squeeze(-1) == 0)
        is_dc = (hit_type.squeeze(-1) == 1)

        mv = torch.zeros(features.shape[0], self.num_blades,
                         device=features.device, dtype=features.dtype)

        # Wire geometry is common to both algebras; only what it becomes differs.
        drift_normed = drift_geometry = wire_normed = wire_dir = None
        if is_dc.any():
            dc = features[is_dc]
            # M3: isotropic norm for wire positions
            wire_normed = dc[:, 4:7] / self.pos_scale
            # The legacy representation deliberately amplifies a few-mm drift
            # radius relative to metre-scale positions. Physical mode instead
            # uses one length unit for every geometric coordinate; the
            # separately normalized scalar below keeps the small radius visible.
            drift_normed = dc[:, 7] / 5.0
            drift_geometry = (
                dc[:, 7] / self.pos_scale
                if self.physical_drift_geometry
                else drift_normed
            )
            cos_s = torch.cos(dc[:, 9])
            sin_s = torch.sin(dc[:, 9])
            cos_a = torch.cos(dc[:, 8])
            sin_a = torch.sin(dc[:, 8])
            if self.fix_wire_dir:
                wire_dir = torch.stack(
                    [sin_s * sin_a, -sin_s * cos_a, cos_s], dim=-1)
            else:
                wire_dir = torch.stack(
                    [sin_s * cos_a, sin_s * sin_a, cos_s], dim=-1)
            wire_dir = wire_dir / (torch.norm(wire_dir, dim=-1, keepdim=True) + 1e-8)

        if self.projective and self.pga_encoding in ("ggtf", "ggtf_wire"):
            # GGTF's encoding: every hit is a point plus a translation. A vertex
            # hit translates by nothing; a drift hit sits at one tangency point
            # and translates to the other, so the drift radius survives as half
            # the length of that translation rather than as a scalar.
            #
            # The drift direction comes straight from the data: our parquet
            # carries the same left and right tangency points GGTF reads off the
            # DD4hep cell frame, and right - left is exactly 2 * drift_distance
            # long with its midpoint exactly on the wire. Earlier this arm
            # reconstructed the axis as wire_dir x r_hat, which is off by a
            # median 48 degrees and wrong by more than 10 degrees for 94% of
            # hits; that would have handicapped the projective baseline in
            # precisely the place we claim it loses, which is the one direction
            # the comparison must not err in.
            #
            # The offset uses drift_normed, which is the drift on its own /5
            # scale rather than the /pos_scale the positions use, so the two
            # tangency points come out far further apart than the physical few
            # millimetres. That is deliberate. The conformal arm inflates the
            # same quantity by the same factor -- its drift sphere has radius
            # drift_normed too -- and matching the inflation keeps the drift
            # signal the same size in both arms, so the comparison isolates the
            # algebra rather than the feature scaling. It also errs in the
            # projective arm's favour, which is the direction we want when the
            # claim is that this arm loses.
            point = pos_normed
            vec = torch.zeros_like(pos_normed)
            if is_dc.any():
                off = 11 if self.use_time else 10
                if features.shape[1] < off + 3:
                    raise ValueError(
                        "pga_hit_encoding='ggtf' needs the drift-direction "
                        "columns; construct the dataset with with_drift_dir=True")
                u = features[is_dc, off:off + 3]
                u = u / (torch.norm(u, dim=-1, keepdim=True) + 1e-8)
                offset = drift_geometry.unsqueeze(-1) * u
                point = point.clone()
                point[is_dc] = wire_normed - offset
                vec[is_dc] = 2.0 * offset
            homog = torch.cat([point, torch.ones_like(point[:, :1])], dim=-1)
            trans = torch.cat([vec, torch.ones_like(vec[:, :1])], dim=-1)
            mv = ((homog @ self.point_matrix.T)
                  + (trans @ self.translation_matrix.T)).to(mv.dtype)
            if self.pga_encoding == "ggtf_wire":
                # The control for M39. GGTF's pair fixes the wire position and
                # the radius but leaves the wire direction underdetermined, so
                # hand it over explicitly in a second channel. PGA still cannot
                # hold the circle as one object; it can hold both facts at once
                # across two. If conformal beats this, the claim is not that
                # the projective arm was starved of information.
                second = torch.zeros_like(mv)
                if is_dc.any():
                    moment = torch.cross(wire_normed, wire_dir, dim=-1)
                    plucker = torch.cat([wire_dir, moment], dim=-1)
                    second[is_dc] = (plucker @ self.line_matrix.T).to(mv.dtype)
                mv = torch.stack([mv, second], dim=1)
        elif self.projective and self.pga_encoding == "point_line":
            # Information-matched, gauge-free PGA control. Channel 0 carries
            # the measured point on the wire (or the VTX point); channel 1
            # carries the full Pluecker wire. Drift radius is a scalar below.
            mv = torch.zeros(features.shape[0], 2, self.num_blades,
                             device=features.device, dtype=features.dtype)
            if is_vtx.any():
                p = pos_normed[is_vtx]
                homog = torch.cat([p, torch.ones_like(p[:, :1])], dim=-1)
                mv[is_vtx, 0] = (homog @ self.point_matrix.T).to(mv.dtype)
            if is_dc.any():
                homog = torch.cat(
                    [wire_normed, torch.ones_like(wire_normed[:, :1])], dim=-1)
                mv[is_dc, 0] = (homog @ self.point_matrix.T).to(mv.dtype)
                moment = torch.cross(wire_normed, wire_dir, dim=-1)
                plucker = torch.cat([wire_dir, moment], dim=-1)
                mv[is_dc, 1] = (plucker @ self.line_matrix.T).to(mv.dtype)
        elif self.projective:
            # Points are trivectors and the wire is a line bivector in Plucker
            # form. The drift radius has no geometric home here and is handed to
            # the scalar channel below — the limitation under test.
            if is_vtx.any():
                p = pos_normed[is_vtx]
                homog = torch.cat([p, torch.ones_like(p[:, :1])], dim=-1)
                mv[is_vtx] = (homog @ self.point_matrix.T).to(mv.dtype)
            if is_dc.any():
                moment = torch.cross(wire_normed, wire_dir, dim=-1)
                plucker = torch.cat([wire_dir, moment], dim=-1)
                mv[is_dc] = (plucker @ self.line_matrix.T).to(mv.dtype)
        elif self.cga_encoding == "point_line":
            # Paper-faithful, deployable drift observation. Channel 0 is always
            # a CGA point: the VTX position or the digitized point on the wire.
            # Channel 1 is the complete conformal wire line for DC hits and zero
            # for VTX hits. The measured drift radius and hit type are auxiliary
            # scalars below. This specifies the tube-like measurement without
            # pretending that the noisy longitudinal coordinate defines an
            # exact circle plane.
            mv = torch.zeros(
                features.shape[0], 2, self.num_blades,
                device=features.device, dtype=features.dtype,
            )
            if is_vtx.any():
                mv[is_vtx, 0] = embed_point(
                    pos_normed[is_vtx]
                ).to(mv.dtype)
            if is_dc.any():
                mv[is_dc, 0] = embed_point(wire_normed).to(mv.dtype)
                mv[is_dc, 1] = embed_line(
                    wire_normed, wire_dir, self.basis_outer
                ).to(mv.dtype)
        elif self.cga_encoding in ("sphere_circle", "sphere_plane"):
            # Channel 0 is grade-1 for both hit types -- a null point for a
            # vertex hit, a drift sphere for a wire hit -- so their inner product
            # is the point-to-sphere distance. Channel 1 is the grade-2 circle
            # for a wire hit and zero for a vertex hit, which has no second
            # constraint to carry.
            mv = torch.zeros(features.shape[0], 2, self.num_blades,
                             device=features.device, dtype=features.dtype)
            if is_vtx.any():
                mv[is_vtx, 0] = embed_point(pos_normed[is_vtx]).to(mv.dtype)
            if is_dc.any():
                mv[is_dc, 0] = embed_sphere(
                    wire_normed, drift_geometry, fix_null=self.fix_cga_null
                ).to(mv.dtype)
                if self.cga_encoding == "sphere_circle":
                    mv[is_dc, 1] = embed_circle_ipns(
                        wire_normed, wire_dir, drift_geometry, self.basis_outer,
                        fix_null=self.fix_cga_null
                    ).to(mv.dtype)
                else:
                    mv[is_dc, 1] = embed_plane(
                        wire_dir, wire_normed, fix_null=self.fix_cga_null
                    ).to(mv.dtype)
        else:
            if is_vtx.any():
                mv[is_vtx] = embed_point(pos_normed[is_vtx]).to(mv.dtype)
            if is_dc.any():
                mv[is_dc] = embed_circle_ipns(
                    wire_normed, wire_dir, drift_geometry, self.basis_outer,
                    fix_null=self.fix_cga_null
                ).to(mv.dtype)

        # Same as adding embed_scalar(hit_type); written by index so it does not
        # assume 32 blades. In two-channel mode it goes on the shared grade-1
        # channel, where the single-channel arm also puts it.
        if not self.separate_hit_type:
            if mv.dim() == 3:
                mv[:, 0, 0] = mv[:, 0, 0] + hit_type.squeeze(-1)
            else:
                mv[:, 0] = mv[:, 0] + hit_type.squeeze(-1)

        if self.args.normalize_mv_inputs:
            # Euclidean per-hit normalization. This is ROTATION-equivariant (a
            # rotation acts as a real orthogonal map on the 32 components, so the
            # L2 norm is preserved) — which is the symmetry that matters for
            # tracking. It is deliberately NOT translation-equivariant: the IP and
            # detector sit at fixed positions, so absolute position is meaningful
            # and translation is not a physical symmetry here. The equivariant CGA
            # grade-wise norm cannot be used on the inputs because VTX hits are
            # null vectors (<P,P> = 0), so their CGA norm is identically zero.
            # Per channel, so a vertex hit's empty circle channel stays exactly
            # zero rather than being blown up to unit norm.
            mv_norm = torch.norm(mv, dim=-1, keepdim=True).clamp(min=1e-6)
            mv = mv / mv_norm

        scalar_parts = []
        if (self.separate_hit_metadata
                or (self.projective
                    and self.pga_encoding in ("line", "point_line"))):
            drift = torch.zeros(features.shape[0], 1,
                                device=features.device, dtype=features.dtype)
            if is_dc.any():
                drift[is_dc, 0] = drift_normed
            scalar_parts.append(drift)
        if self.separate_hit_type:
            scalar_parts.append(hit_type)
        if self.use_time:
            scalar_parts.append(features[:, 10:11])
        scalars = torch.cat(scalar_parts, dim=-1) if scalar_parts else None

        return mv, scalars

    def _backbone_outputs(self, mv, scalars, seq_lens):
        """Run the equivariant stack and retain both output representations."""
        if mv.dim() == 2:
            mv = mv.unsqueeze(1)  # (N, 1, blades) — single channel
        # else the embedding already produced (N, channels, blades)

        # Pass per-event hit counts rather than a dense (1,1,M,M) bool mask.
        # The attention primitive runs block-diagonal attention as independent
        # per-event SDPA, so it never allocates the O(M^2) score matrix (the big
        # VRAM cost under the MATH backend that the large MV feature dim forces).
        # A dense bool mask is still accepted by the primitive for back-compat;
        # `block_diagonal_bool_mask` remains available for that path.
        #
        # The packed seq_lens form dispatches to xformers, which is CUDA-only;
        # off GPU fall back to the equivalent dense mask so the model stays
        # runnable on CPU for smoke tests and debugging.
        if mv.is_cuda:
            attn = list(seq_lens)
        else:
            attn = block_diagonal_bool_mask(seq_lens, device=mv.device)

        return self.cgatr(mv, scalars=scalars, attention_mask=attn)

    def backbone(self, mv, scalars, seq_lens):
        """The equivariant stack. Takes and returns one multivector per hit."""
        out_mv, _ = self._backbone_outputs(mv, scalars, seq_lens)
        return out_mv[:, 0, :]


# ---------------------------------------------------------------------------
# Object condensation loss — torch_scatter port of hgcalimplementation
# with selectable repulsive potential (no DGL dependency)
# ---------------------------------------------------------------------------
def object_condensation_loss(
    coords, beta, mc_index, batch,
    noise_index=0, qmin=0.1,
    attr_weight=1.0, repul_weight=1.0, fill_loss_weight=0.0,
    use_average_cc_pos=0.0, s_B=1.0,
    beta_suppress_weight=0.0,
    var_weight=0.0,
    return_components=False,
    detach_components=True,
    oc_mode="paper_hinge",
    track_separation_weight=None,
):
    """Hybrid object-condensation loss using torch_scatter.

    Both modes use the logarithmic attractive potential and per-object
    normalization from the HGCAL/GGTF implementation. ``paper_hinge`` combines
    that attraction with Kieseler's compact-support hinge repulsion and the
    original arctanh-squared charge. It is the empirically selected tracking
    objective, but it is not the literal Kieseler loss (whose attraction is
    quadratic). ``ggtf`` instead uses GGTF's Gaussian repulsion and softened
    charge, optionally with inverse nearest-track-separation weights.

    Matches the original calc_LV_Lbeta semantics:
    - Attraction: signal hits pulled toward their own condensation point
    - Repulsion: ALL hits (incl. noise) pushed away from non-own-cluster objects
    - Repulsion computed per-event to control memory
    - Beta suppression: penalizes high beta for non-alpha signal hits

    v35 additions:
    - Within-cluster variance regularizer L_var = mean_k mean_i ||x_i - mu_k||^2
      where mu_k is the mean of signal-hit coords assigned to track k.
    - If return_components=True, returns (total, dict of components).
    """
    device = coords.device
    beta = torch.nan_to_num(beta, nan=0.0)

    is_noise = mc_index == noise_index
    is_sig = ~is_noise

    n_hits = coords.shape[0]
    n_hits_sig = is_sig.sum().item()
    if n_hits_sig < 4:
        return torch.tensor(0.0, device=device, requires_grad=True)

    sig_coords = coords[is_sig]
    sig_beta = beta[is_sig]
    sig_mc = mc_index[is_sig]
    sig_batch = batch[is_sig]

    # Per-event reincrementalization of signal labels -> contiguous 0..K_e-1
    object_index = torch.empty_like(sig_mc)
    n_objects_per_event_list = []
    unique_events = sig_batch.unique()
    for evt in unique_events:
        evt_mask = sig_batch == evt
        _, inv = sig_mc[evt_mask].unique(return_inverse=True)
        object_index[evt_mask] = inv
        n_objects_per_event_list.append(inv.max().item() + 1)

    n_objects_per_event = torch.tensor(n_objects_per_event_list, device=device, dtype=torch.long)

    # Make object_index globally unique across events
    offsets = torch.zeros_like(n_objects_per_event)
    offsets[1:] = n_objects_per_event[:-1].cumsum(dim=0)
    _, event_remap = sig_batch.unique(return_inverse=True)
    object_index = object_index + offsets[event_remap]

    n_objects = n_objects_per_event.sum().item()
    if n_objects < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    if oc_mode not in ("paper_hinge", "ggtf"):
        raise ValueError(f"Unknown object-condensation mode: {oc_mode}")

    # q for ALL hits (repulsion uses noise hits too, matching original).
    # GGTF's HGCAL-derived implementation divides by 1.01; the paper mode does not.
    q_scale = 1.01 if oc_mode == "ggtf" else 1.0
    q_all = (beta.clip(0.0, 1 - 1e-4).arctanh() / q_scale) ** 2 + qmin
    q_sig = q_all[is_sig]

    # Alpha points (condensation points)
    q_alpha, index_alpha = scatter_max(q_sig, object_index)
    x_alpha = sig_coords[index_alpha]
    beta_alpha = sig_beta[index_alpha]
    object_repulsion_weight = None
    if track_separation_weight is not None:
        if track_separation_weight.shape != beta.shape:
            raise ValueError(
                "track_separation_weight must have one value per input hit"
            )
        object_repulsion_weight = scatter_mean(
            track_separation_weight[is_sig].float(), object_index
        )

    # --- Attractive potential (signal hits only, per-hit) ---
    e1 = torch.exp(torch.tensor(1.0, device=device))
    d_sq_own = ((sig_coords - x_alpha[object_index]) ** 2).sum(dim=1)
    norms_att = torch.log(e1 * d_sq_own / 2 + 1)
    V_att_per_hit = q_sig * q_alpha[object_index] * norms_att

    V_att_per_obj = scatter_add(V_att_per_hit, object_index)
    n_hits_per_obj = scatter_add(torch.ones(n_hits_sig, device=device), object_index)
    V_att_per_obj = V_att_per_obj / (n_hits_per_obj + 1e-3)
    L_V_att = V_att_per_obj.mean()

    # --- Within-cluster variance regularizer (v35) ---
    x_centroid = scatter_mean(sig_coords, object_index, dim=0)
    d_sq_centroid = ((sig_coords - x_centroid[object_index]) ** 2).sum(dim=1)
    L_var_per_obj = scatter_mean(d_sq_centroid, object_index)
    L_var = L_var_per_obj.mean()

    # --- Repulsive potential (per-event, ALL hits incl. noise, matching original) ---
    all_object_index = torch.full((n_hits,), -1, device=device, dtype=torch.long)
    all_object_index[is_sig] = object_index

    rep_sum = torch.tensor(0.0, device=device)
    rep_normalization = torch.tensor(0.0, device=device)
    obj_offset = 0

    for i, evt_val in enumerate(unique_events):
        n_evt_obj = n_objects_per_event[i].item()
        if n_evt_obj < 2:
            obj_offset += n_evt_obj
            continue

        evt_mask = batch == evt_val
        evt_coords = coords[evt_mask]
        evt_q = q_all[evt_mask]
        evt_obj = all_object_index[evt_mask]

        evt_x_alpha = x_alpha[obj_offset:obj_offset + n_evt_obj]
        evt_q_alpha = q_alpha[obj_offset:obj_offset + n_evt_obj]

        d_sq = ((evt_coords.unsqueeze(1) - evt_x_alpha.unsqueeze(0)) ** 2).sum(-1)
        if oc_mode == "ggtf":
            # GGTF/HGCAL tracking path: Gaussian in squared latent distance.
            exp_rep = torch.exp(-d_sq / 2.0)
        else:
            # Kieseler 2002.03605: compact-support linear hinge.
            exp_rep = torch.relu(1.0 - torch.sqrt(d_sq.clamp(min=1e-12)))

        local_obj = evt_obj.clone()
        has_obj = local_obj >= 0
        local_obj[has_obj] -= obj_offset
        own_mask = torch.zeros(evt_coords.shape[0], n_evt_obj, device=device)
        if has_obj.any():
            own_mask[has_obj] = torch.nn.functional.one_hot(
                local_obj[has_obj], num_classes=n_evt_obj
            ).float()
        M_inv = 1.0 - own_mask

        V_rep = evt_q.unsqueeze(1) * evt_q_alpha.unsqueeze(0) * exp_rep * M_inv
        V_rep_per_obj = V_rep.sum(dim=0)
        n_rep_terms = M_inv.sum(dim=0).clamp(min=1.0)
        V_rep_per_obj = V_rep_per_obj / n_rep_terms

        if object_repulsion_weight is None:
            rep_sum = rep_sum + V_rep_per_obj.sum()
            rep_normalization = rep_normalization + n_evt_obj
        else:
            evt_weight = object_repulsion_weight[
                obj_offset:obj_offset + n_evt_obj
            ]
            rep_sum = rep_sum + (V_rep_per_obj * evt_weight).sum()
            rep_normalization = rep_normalization + evt_weight.sum()
        obj_offset += n_evt_obj

    L_V_rep = rep_sum / rep_normalization.clamp(min=1.0)
    L_V = attr_weight * L_V_att + repul_weight * L_V_rep

    # --- L_beta signal ---
    beta_sum_per_obj = scatter_add(sig_beta, object_index)
    L_beta_sig = torch.mean(1 - beta_alpha + 1 - torch.clip(beta_sum_per_obj, 0, 1))

    # --- L_beta noise (matches original: .sum() / batch_size) ---
    batch_size = batch.unique().numel()
    L_beta_noise = torch.tensor(0.0, device=device)
    if is_noise.any():
        noise_beta = beta[is_noise]
        noise_batch = batch[is_noise]
        _, noise_evt_remap = noise_batch.unique(return_inverse=True)
        n_noise_per_evt = scatter_add(
            torch.ones_like(noise_evt_remap, dtype=torch.float), noise_evt_remap
        ).clamp(min=1.0)
        beta_noise_per_evt = scatter_add(noise_beta, noise_evt_remap)
        L_beta_noise = s_B * (beta_noise_per_evt / n_noise_per_evt).sum() / batch_size

    # --- Beta suppression: push non-alpha signal betas toward 0 ---
    L_beta_suppress = torch.tensor(0.0, device=device)
    if beta_suppress_weight > 0 and n_hits_sig > n_objects:
        is_alpha = torch.zeros(n_hits_sig, dtype=torch.bool, device=device)
        is_alpha[index_alpha] = True
        L_beta_suppress = beta_suppress_weight * sig_beta[~is_alpha].mean()

    total = L_V + L_beta_sig + L_beta_noise + L_beta_suppress + var_weight * L_var
    if return_components:
        def component(value):
            if not torch.is_tensor(value):
                value = torch.tensor(float(value), device=device)
            return value.detach() if detach_components else value

        components = {
            "L_V_att": component(L_V_att),
            "L_V_rep": component(L_V_rep),
            "L_beta_sig": component(L_beta_sig),
            "L_beta_noise": component(L_beta_noise),
            "L_beta_suppress": component(L_beta_suppress),
            "L_var": component(L_var),
            "var_weight": torch.tensor(float(var_weight), device=device),
        }
        return total, components
    return total


# ---------------------------------------------------------------------------
# Beta-greedy evaluation (pure numpy)
# ---------------------------------------------------------------------------
def get_clustering_np(betas, X, tbeta=0.5, td=0.5):
    n_points = betas.shape[0]
    select = betas > tbeta
    indices = np.nonzero(select)[0]
    indices = indices[np.argsort(-betas[select])]
    unassigned = np.arange(n_points)
    clustering = -1 * np.ones(n_points, dtype=np.int32)
    for idx in indices:
        d = np.linalg.norm(X[unassigned] - X[idx], axis=-1)
        assigned = unassigned[d < td]
        clustering[assigned] = idx
        unassigned = unassigned[~(d < td)]
    return clustering


def _seq_lens_to_batch(seq_lens, device):
    return torch.repeat_interleave(
        torch.arange(len(seq_lens), device=device, dtype=torch.long),
        torch.tensor(seq_lens, device=device, dtype=torch.long),
    )


def _compute_var_weight(epoch, args):
    """Linear warmup of var_weight from 0 to args.var_weight over the first
    `args.var_warmup_epochs` epochs (inclusive of the final epoch)."""
    if args.var_warmup_epochs <= 0:
        return float(args.var_weight)
    # epoch is 1-indexed in this codebase
    frac = min(1.0, max(0.0, (epoch - 1) / max(args.var_warmup_epochs, 1)))
    return float(args.var_weight) * frac


def _compute_batch_metrics_greedy(coords, beta_logits, mc_index, is_secondary, seq_lens, tbeta=0.5, td=0.5, cosine_norm=False, noise_index=0):
    from collections import Counter

    coords = coords.detach().cpu().numpy()
    beta = torch.sigmoid(beta_logits.squeeze(-1)).detach().cpu().numpy()
    mc_index = mc_index.detach().cpu().numpy()
    is_secondary = is_secondary.detach().cpu().numpy()

    all_purities, all_effs, all_match = [], [], []
    all_match_strict50 = []  # purity > 0.75 AND efficiency_per_hit >= 0.5
    noise_low_beta_counts, noise_total_counts = 0, 0
    offset = 0
    for n_hits in seq_lens:
        sl = slice(offset, offset + n_hits)
        offset += n_hits

        sec_mask = is_secondary[sl] | (mc_index[sl] == noise_index)
        if sec_mask.any():
            noise_total_counts += sec_mask.sum()
            noise_low_beta_counts += (beta[sl][sec_mask] < 0.1).sum()

        mask = (~is_secondary[sl]) & (mc_index[sl] != noise_index)
        c = coords[sl][mask]
        if cosine_norm:
            norms = np.linalg.norm(c, axis=1, keepdims=True)
            c = c / np.clip(norms, 1e-6, None)
        b = beta[sl][mask]
        mc = mc_index[sl][mask]
        if len(c) == 0:
            continue

        pred_labels = get_clustering_np(b, c, tbeta=tbeta, td=td)
        unique_pred = np.unique(pred_labels[pred_labels >= 0])
        unique_true = np.unique(mc[mc >= 0])
        if len(unique_pred) == 0:
            all_purities.append(0.0)
            all_effs.append(0.0)
            all_match.append(0.0)
            continue

        purities = []
        for pid in unique_pred:
            cluster_mc = mc[pred_labels == pid]
            if len(cluster_mc) > 0:
                purities.append(np.bincount(cluster_mc).max() / len(cluster_mc))
        if purities:
            all_purities.append(np.mean(purities))

        matched = 0
        matched_strict50 = 0
        n_matchable = 0
        effs = []
        for tid in unique_true:
            tmask = mc == tid
            n_true = tmask.sum()
            if n_true < 2:
                continue
            n_matchable += 1
            pred_for_track = pred_labels[tmask]
            pred_for_track = pred_for_track[pred_for_track >= 0]
            if len(pred_for_track) == 0:
                effs.append(0.0)
                continue
            best_label, best_match = Counter(pred_for_track).most_common(1)[0]
            eff = best_match / n_true
            pur = best_match / (pred_labels == best_label).sum()
            effs.append(eff)
            if pur > 0.75:
                matched += 1
                if eff >= 0.5:
                    matched_strict50 += 1
        if effs:
            all_effs.append(np.mean(effs))
            all_match.append(matched / max(n_matchable, 1))
            all_match_strict50.append(matched_strict50 / max(n_matchable, 1))

    noise_suppression = float(noise_low_beta_counts / max(noise_total_counts, 1))

    return {
        "purity": float(np.mean(all_purities)) if all_purities else 0.0,
        "efficiency": float(np.mean(all_effs)) if all_effs else 0.0,
        "match_rate": float(np.mean(all_match)) if all_match else 0.0,
        "match_rate_strict50": float(np.mean(all_match_strict50)) if all_match_strict50 else 0.0,
        "noise_suppression": noise_suppression,
    }
