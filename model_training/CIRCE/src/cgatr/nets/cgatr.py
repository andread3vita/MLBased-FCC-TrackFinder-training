"""CGATr: Conformal Geometric Algebra Transformer for Cl(4,1).

Main architecture with 32-component multivectors.
"""

from dataclasses import replace
from functools import partial
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.cgatr.layers.attention.config import SelfAttentionConfig
from src.cgatr.layers.gatr_block import CGATrBlock
from src.cgatr.layers.linear import EquiLinear
from src.cgatr.layers.mlp.config import MLPConfig


class CGATr(nn.Module):
    """CGATr network for data with a single token dimension.

    Architecture: linear_in -> N x CGATrBlock -> linear_out
    Uses 32-dim CGA multivectors with Pin(4,1) equivariance.

    Parameters
    ----------
    in_mv_channels : int
    out_mv_channels : int
    hidden_mv_channels : int
    in_s_channels : int or None
    out_s_channels : int or None
    hidden_s_channels : int or None
    attention : SelfAttentionConfig
    mlp : MLPConfig
    basis_gp : (32, 32, 32) GP Cayley table
    basis_ip_weights : (32,) inner product weights
    basis_outer : (32, 32, 32) outer product table
    basis_pin : (9, 32, 32) equivariant linear basis
    basis_q, basis_k : (5, 5, 6) distance basis
    num_blocks : int
    dropout_prob : float or None
    """

    def __init__(
        self,
        in_mv_channels: int,
        out_mv_channels: int,
        hidden_mv_channels: int,
        in_s_channels: Optional[int],
        out_s_channels: Optional[int],
        hidden_s_channels: Optional[int],
        attention: SelfAttentionConfig,
        basis_gp,
        basis_ip_weights,
        basis_outer,
        basis_pin,
        basis_q,
        basis_k,
        mlp: MLPConfig,
        num_blocks: int = 8,
        reinsert_mv_channels: Optional[Tuple[int]] = None,
        reinsert_s_channels: Optional[Tuple[int]] = None,
        checkpoint_blocks: bool = False,
        dropout_prob: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.linear_in = EquiLinear(
            basis_pin=basis_pin,
            in_mv_channels=in_mv_channels,
            out_mv_channels=hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=hidden_s_channels,
        )

        attention = replace(
            SelfAttentionConfig.cast(attention),
            additional_qk_mv_channels=0
            if reinsert_mv_channels is None
            else len(reinsert_mv_channels),
            additional_qk_s_channels=0
            if reinsert_s_channels is None
            else len(reinsert_s_channels),
        )
        mlp = MLPConfig.cast(mlp)

        self.blocks = nn.ModuleList([
            CGATrBlock(
                gp=basis_gp,
                ip_weights=basis_ip_weights,
                outer=basis_outer,
                basis_pin=basis_pin,
                basis_q=basis_q,
                basis_k=basis_k,
                mv_channels=hidden_mv_channels,
                s_channels=hidden_s_channels,
                attention=attention,
                mlp=mlp,
                dropout_prob=dropout_prob,
            )
            for _ in range(num_blocks)
        ])

        self.linear_out = EquiLinear(
            basis_pin=basis_pin,
            in_mv_channels=hidden_mv_channels,
            out_mv_channels=out_mv_channels,
            in_s_channels=hidden_s_channels,
            out_s_channels=out_s_channels,
        )

        self._reinsert_s_channels = reinsert_s_channels
        self._reinsert_mv_channels = reinsert_mv_channels
        self._checkpoint_blocks = checkpoint_blocks
        self.basis_pin = basis_pin

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Forward pass.

        Parameters
        ----------
        multivectors : torch.Tensor (..., items, in_mv_channels, 32)
        scalars : torch.Tensor (..., items, in_s_channels) or None
        attention_mask : optional

        Returns
        -------
        outputs_mv : torch.Tensor (..., items, out_mv_channels, 32)
        outputs_s : torch.Tensor or None
        """
        reference_mv = self._construct_dual_reference(multivectors)
        additional_qk_features_mv = None
        additional_qk_features_s = None

        h_mv, h_s = self.linear_in(multivectors, scalars=scalars)

        for block in self.blocks:
            if self._checkpoint_blocks and self.training:
                # Gradient checkpointing: recompute activations during backward
                # Saves ~70% VRAM at cost of ~30% slower training
                # Bind non-tensor args (BlockDiagonalMask, None) via partial
                # to avoid DDP deadlock — only tensors should flow through checkpoint()
                block_fn = partial(
                    block,
                    reference_mv=reference_mv,
                    additional_qk_features_mv=additional_qk_features_mv,
                    additional_qk_features_s=additional_qk_features_s,
                    attention_mask=attention_mask,
                )
                h_mv, h_s = checkpoint(
                    block_fn, h_mv, h_s,
                    use_reentrant=False,
                )
            else:
                h_mv, h_s = block(
                    h_mv,
                    scalars=h_s,
                    reference_mv=reference_mv,
                    additional_qk_features_mv=additional_qk_features_mv,
                    additional_qk_features_s=additional_qk_features_s,
                    attention_mask=attention_mask,
                )

        outputs_mv, outputs_s = self.linear_out(h_mv, scalars=h_s)
        return outputs_mv, outputs_s

    @staticmethod
    def _construct_dual_reference(inputs: torch.Tensor) -> torch.Tensor:
        """Construct reference multivector for equivariant join from input mean.

        For input shape (N_items, channels, 32), averages over items AND channels
        to get a single reference (1, 1, 32) that broadcasts with any channel count.
        """
        # Average over all dims except the last (MV components)
        # Input: (N, channels, 32) -> mean over N and channels -> (1, 1, 32)
        mean_dim = tuple(range(0, len(inputs.shape) - 1))
        return torch.mean(inputs, dim=mean_dim, keepdim=True)
