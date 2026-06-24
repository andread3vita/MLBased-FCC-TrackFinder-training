"""Pin(4,1)-equivariant linear layers for CGA with 32-dim multivectors."""

from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from src.cgatr.interface import embed_scalar
from src.cgatr.primitives.linear import NUM_PIN_LINEAR_BASIS_ELEMENTS, equi_linear


class EquiLinear(nn.Module):
    """Pin(4,1)-equivariant linear layer for 32-dim multivectors.

    Maps (..., in_channels, 32) -> (..., out_channels, 32) using 9 basis elements.

    Parameters
    ----------
    basis_pin : torch.Tensor with shape (9, 32, 32)
    in_mv_channels : int
    out_mv_channels : int
    in_s_channels : int or None
    out_s_channels : int or None
    bias : bool
    initialization : str
    """

    def __init__(
        self,
        basis_pin,
        in_mv_channels: int,
        out_mv_channels: int,
        in_s_channels: Optional[int] = None,
        out_s_channels: Optional[int] = None,
        bias: bool = True,
        initialization: str = "default",
    ) -> None:
        super().__init__()
        self.register_buffer("basis", basis_pin)
        self._in_mv_channels = in_mv_channels

        if initialization == "unit_scalar":
            assert bias
            if in_s_channels is None:
                raise NotImplementedError(
                    "unit_scalar initialization requires scalar inputs"
                )

        # MV -> MV weights: (out, in, 9)
        self.weight = nn.Parameter(
            torch.empty(
                (out_mv_channels, in_mv_channels, NUM_PIN_LINEAR_BASIS_ELEMENTS)
            )
        )

        self.bias = (
            nn.Parameter(torch.zeros((out_mv_channels, 1)))
            if bias and in_s_channels is None
            else None
        )

        # Scalars -> MV scalars
        self.s2mvs: Optional[nn.Linear]
        if in_s_channels:
            self.s2mvs = nn.Linear(in_s_channels, out_mv_channels, bias=bias)
        else:
            self.s2mvs = None

        # MV scalars -> scalars
        if out_s_channels:
            self.mvs2s = nn.Linear(in_mv_channels, out_s_channels, bias=bias)
        else:
            self.mvs2s = None

        # Scalars -> scalars
        if in_s_channels is not None and out_s_channels is not None:
            self.s2s = nn.Linear(in_s_channels, out_s_channels, bias=False)
        else:
            self.s2s = None

        self.reset_parameters(initialization)

    def forward(
        self, multivectors: torch.Tensor, scalars: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Forward pass.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., in_mv_channels, 32)
        scalars : None or torch.Tensor with shape (..., in_s_channels)

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., out_mv_channels, 32)
        outputs_s : None or torch.Tensor with shape (..., out_s_channels)
        """
        outputs_mv = equi_linear(self.basis, multivectors, self.weight)

        if self.bias is not None:
            bias = embed_scalar(self.bias)
            outputs_mv = outputs_mv + bias

        if self.s2mvs is not None and scalars is not None:
            outputs_mv[..., 0] += self.s2mvs(scalars)

        if self.mvs2s is not None:
            outputs_s = self.mvs2s(multivectors[..., 0])
            if self.s2s is not None and scalars is not None:
                outputs_s = outputs_s + self.s2s(scalars)
        else:
            outputs_s = None

        return outputs_mv, outputs_s

    def reset_parameters(
        self,
        initialization: str,
        gain: float = 1.0,
        additional_factor=1.0 / np.sqrt(3.0),
        use_mv_heuristics=True,
    ) -> None:
        """Initialize weights."""
        mv_component_factors, mv_factor, mvs_bias_shift, s_factor = (
            self._compute_init_factors(
                initialization, gain, additional_factor, use_mv_heuristics
            )
        )
        self._init_multivectors(mv_component_factors, mv_factor, mvs_bias_shift)
        self._init_scalars(s_factor)

    @staticmethod
    def _compute_init_factors(
        initialization, gain, additional_factor, use_mv_heuristics
    ):
        if initialization == "default":
            mv_factor = gain * additional_factor * np.sqrt(3)
            s_factor = gain * additional_factor * np.sqrt(3)
            mvs_bias_shift = 0.0
        elif initialization == "small":
            mv_factor = 0.1 * gain * additional_factor * np.sqrt(3)
            s_factor = 0.1 * gain * additional_factor * np.sqrt(3)
            mvs_bias_shift = 0.0
        elif initialization == "unit_scalar":
            mv_factor = 0.1 * gain * additional_factor * np.sqrt(3)
            s_factor = gain * additional_factor * np.sqrt(3)
            mvs_bias_shift = 1.0
        elif initialization == "almost_unit_scalar":
            mv_factor = 0.5 * gain * additional_factor * np.sqrt(3)
            s_factor = gain * additional_factor * np.sqrt(3)
            mvs_bias_shift = 1.0
        else:
            raise ValueError(f"Unknown initialization: {initialization}")

        if use_mv_heuristics:
            # Heuristic correction factors for 9 CGA basis elements
            # 6 grade projections + 3 cross-grade maps
            mv_component_factors = torch.sqrt(
                torch.Tensor([1.0, 5.0, 10.0, 10.0, 5.0, 1.0, 0.5, 1.0, 0.5])
            )
        else:
            mv_component_factors = torch.ones(NUM_PIN_LINEAR_BASIS_ELEMENTS)

        return mv_component_factors, mv_factor, mvs_bias_shift, s_factor

    def _init_multivectors(self, mv_component_factors, mv_factor, mvs_bias_shift):
        fan_in = self._in_mv_channels
        bound = mv_factor / np.sqrt(fan_in)
        for i, factor in enumerate(mv_component_factors):
            nn.init.uniform_(self.weight[..., i], a=-factor * bound, b=factor * bound)

        if self.s2mvs is not None:
            bound = mv_component_factors[0] * mv_factor / np.sqrt(fan_in) / np.sqrt(2)
            nn.init.uniform_(self.weight[..., [0]], a=-bound, b=bound)

        if self.s2mvs is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.s2mvs.weight)
            fan_in = max(fan_in, 1)
            bound = mv_component_factors[0] * mv_factor / np.sqrt(fan_in) / np.sqrt(2)
            nn.init.uniform_(self.s2mvs.weight, a=-bound, b=bound)

            if self.s2mvs.bias is not None:
                fan_in = (
                    nn.init._calculate_fan_in_and_fan_out(self.s2mvs.weight)[0]
                    + self._in_mv_channels
                )
                bound = mv_component_factors[0] / np.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(
                    self.s2mvs.bias, mvs_bias_shift - bound, mvs_bias_shift + bound
                )

    def _init_scalars(self, s_factor):
        models = []
        if self.s2s:
            models.append(self.s2s)
        if self.mvs2s:
            models.append(self.mvs2s)
        for model in models:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.weight)
            fan_in = max(fan_in, 1)
            bound = s_factor / np.sqrt(fan_in) / np.sqrt(len(models))
            nn.init.uniform_(model.weight, a=-bound, b=bound)
        if self.mvs2s and self.mvs2s.bias is not None:
            fan_in = nn.init._calculate_fan_in_and_fan_out(self.mvs2s.weight)[0]
            if self.s2s:
                fan_in += nn.init._calculate_fan_in_and_fan_out(self.s2s.weight)[0]
            bound = s_factor / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.mvs2s.bias, -bound, bound)
