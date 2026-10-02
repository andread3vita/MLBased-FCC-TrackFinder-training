"""Pin(4,1)-equivariant linear layers for CGA with 32-dim multivectors."""

from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

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
        # Number of equivariant basis maps is read from the basis itself, so the
        # layer works with both the legacy 9-element basis and the canonical
        # SE(3) null-space basis (~21 elements).
        self._n_basis = int(basis_pin.shape[0])

        if initialization == "unit_scalar":
            assert bias
            if in_s_channels is None:
                raise NotImplementedError(
                    "unit_scalar initialization requires scalar inputs"
                )

        # MV -> MV weights: (out, in, n_basis)
        self.weight = nn.Parameter(
            torch.empty(
                (out_mv_channels, in_mv_channels, self._n_basis)
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

        # Remembered so a model-level switch can re-initialize only the layers
        # that took the generic scheme, leaving the deliberately tuned
        # almost_unit_scalar layers in the bilinears alone.
        self._init_kind = initialization
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
            bias = embed_scalar(self.bias, num_blades=self.basis.shape[-1])
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
        if initialization == "identity_algebra":
            # de Haan et al. (arXiv:2311.04744) App. C report that an
            # initialization that acts as the identity on the algebra is best for
            # C-GATr, while a Kaiming-like one is best for the projective
            # variants. We used Kaiming-like everywhere, which means the
            # conformal arm has been running on the scheme tuned for its
            # competitor and the measured algebra gap is if anything understated.
            self._init_identity_algebra(gain, additional_factor, use_mv_heuristics)
            self._init_scalars(
                self._compute_init_factors("default", gain, additional_factor,
                                           use_mv_heuristics)[3])
            return
        mv_component_factors, mv_factor, mvs_bias_shift, s_factor = (
            self._compute_init_factors(
                initialization, gain, additional_factor, use_mv_heuristics
            )
        )
        self._init_multivectors(mv_component_factors, mv_factor, mvs_bias_shift)
        self._init_scalars(s_factor)

    def _identity_coefficients(self):
        """Coefficients c with sum_i c_i B_i as close to the identity as possible.

        The identity is itself an equivariant map, so it lies in the span of the
        basis and the least-squares residual should be numerically zero; it is
        returned so the caller can assert that rather than assume it.
        """
        basis = self.basis.reshape(self._n_basis, -1).double()
        target = torch.eye(self.basis.shape[-1], dtype=torch.float64,
                           device=basis.device).reshape(-1)
        sol = torch.linalg.lstsq(basis.T, target.unsqueeze(-1)).solution.squeeze(-1)
        resid = (basis.T @ sol - target).norm() / target.norm()
        return sol, float(resid)

    def _init_identity_algebra(self, gain, additional_factor, use_mv_heuristics):
        """Every channel pair starts as a multiple of the identity map.

        The multivector structure is therefore untouched at initialization -- the
        layer is a plain channel mixing -- rather than a random combination of
        all the equivariant maps, which scrambles grades before any training has
        happened.

        Each layer's output scale is matched to the default scheme by measuring
        both on the same random input and rescaling, so the two schemes are not
        confounded by a per-layer gain difference. The composition of many such
        layers still ends up at a different overall scale -- about a quarter of
        the default's on a 2-block conformal stack -- because preserving the
        multivector direction changes how the equivariant LayerNorm and the
        bilinears interact downstream. That is the structural effect under test,
        not a scaling bug, and the trained output head absorbs a global factor.
        """
        c, resid = self._identity_coefficients()
        if resid > 1e-6:
            raise RuntimeError(
                f"the identity is not in the span of the equivariant basis "
                f"(relative residual {resid:.2e}); identity_algebra "
                f"initialization is not well defined for this basis")
        c = c.to(self.weight.dtype)

        gen = torch.Generator(device="cpu").manual_seed(0)
        probe = torch.randn(64, self._in_mv_channels, self.basis.shape[-1],
                            generator=gen, dtype=torch.float32)

        # reference scale from the default scheme
        with torch.no_grad():
            self.reset_parameters("default", gain, additional_factor,
                                  use_mv_heuristics)
            ref = equi_linear(self.basis.float(), probe.to(self.basis.device),
                              self.weight.detach().float()).std()

            chan = torch.empty(self.weight.shape[:2], dtype=self.weight.dtype)
            bound = 1.0 / np.sqrt(self._in_mv_channels)
            nn.init.uniform_(chan, -bound, bound)
            self.weight.copy_(chan.unsqueeze(-1) * c.view(1, 1, -1))

            got = equi_linear(self.basis.float(), probe.to(self.basis.device),
                              self.weight.detach().float()).std()
            self.weight.mul_(ref / got.clamp_min(1e-12))

        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if self.s2mvs is not None:
            fan_in = max(nn.init._calculate_fan_in_and_fan_out(self.s2mvs.weight)[0], 1)
            b = 1.0 / np.sqrt(fan_in)
            nn.init.uniform_(self.s2mvs.weight, -b, b)
            if self.s2mvs.bias is not None:
                nn.init.zeros_(self.s2mvs.bias)

    def _compute_init_factors(
        self, initialization, gain, additional_factor, use_mv_heuristics
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

        if use_mv_heuristics and self._n_basis == NUM_PIN_LINEAR_BASIS_ELEMENTS:
            # Legacy 9-element basis: per-grade heuristic
            # (6 grade projections + 3 cross-grade maps).
            mv_component_factors = torch.sqrt(
                torch.Tensor([1.0, 5.0, 10.0, 10.0, 5.0, 1.0, 0.5, 1.0, 0.5])
            )
        elif use_mv_heuristics:
            # Canonical SE(3) null-space basis (orthonormal maps): spread the
            # legacy total init variance (~34) uniformly across the n_basis maps
            # so the initial output scale matches the tuned 9-element baseline.
            total_var = 34.0
            mv_component_factors = torch.full(
                (self._n_basis,), (total_var / self._n_basis) ** 0.5
            )
        else:
            mv_component_factors = torch.ones(self._n_basis)

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


def fused_equi_linear_same_input(
    modules: Tuple[EquiLinear, ...],
    multivectors: torch.Tensor,
    scalars: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """Evaluate compatible equivariant projections as one larger operation.

    This preserves the original modules and parameter names, so checkpoints and
    optimizer state remain unchanged.  ``torch.cat`` routes gradients back to
    each parameter while the basis materialization, MV contraction, and scalar
    projections use one larger launch instead of one launch per Q/K/V or
    left/right branch.

    The helper intentionally supports only the hidden-block case used by the
    production C-GATr.  Callers fall back to separate modules for any other
    layout.
    """
    if not modules:
        raise ValueError("at least one EquiLinear is required")
    first = modules[0]
    if scalars is None:
        raise ValueError("fused hidden projections require scalar inputs")
    # Do not require identical buffer pointers: Module.cuda() materializes each
    # registered copy of the shared canonical basis in separate CUDA storage.
    if any(
        module.basis.shape != first.basis.shape
        or module.basis.device != first.basis.device
        or module.basis.dtype != first.basis.dtype
        or module._in_mv_channels != first._in_mv_channels
        or module.bias is not None
        or module.s2mvs is None
        or module.s2mvs.in_features != first.s2mvs.in_features
        or (module.mvs2s is None) != (first.mvs2s is None)
        or (
            module.mvs2s is not None
            and module.mvs2s.in_features != first.mvs2s.in_features
        )
        or (module.s2s is None) != (first.s2s is None)
        or (
            module.s2s is not None
            and module.s2s.in_features != first.s2s.in_features
        )
        for module in modules
    ):
        raise ValueError("EquiLinear modules are not fusion-compatible")

    mv_sizes = [module.weight.shape[0] for module in modules]
    weight = torch.cat([module.weight for module in modules], dim=0)
    outputs_mv = equi_linear(first.basis, multivectors, weight)

    s2mv_weight = torch.cat([module.s2mvs.weight for module in modules], dim=0)
    s2mv_bias = torch.cat([module.s2mvs.bias for module in modules], dim=0)
    outputs_mv[..., 0] += F.linear(scalars, s2mv_weight, s2mv_bias)

    mv_parts = outputs_mv.split(mv_sizes, dim=-2)
    if first.mvs2s is None:
        return tuple((part, None) for part in mv_parts)

    s_sizes = [module.mvs2s.weight.shape[0] for module in modules]
    mvs2s_weight = torch.cat([module.mvs2s.weight for module in modules], dim=0)
    mvs2s_bias = torch.cat([module.mvs2s.bias for module in modules], dim=0)
    outputs_s = F.linear(multivectors[..., 0], mvs2s_weight, mvs2s_bias)
    if first.s2s is not None:
        s2s_weight = torch.cat([module.s2s.weight for module in modules], dim=0)
        outputs_s += F.linear(scalars, s2s_weight, bias=None)
    s_parts = outputs_s.split(s_sizes, dim=-1)
    return tuple(zip(mv_parts, s_parts))
