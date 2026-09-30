import torch
import torch.nn as nn


class EMAShadow:
    """Exponential moving average over parameters and normalization buffers."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

    def prepare_for_updates(self):
        """Convert inference-mode shadows to ordinary, mutable tensors."""
        self.shadow = {
            key: value.detach().clone() for key, value in self.shadow.items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        for key, value in model.state_dict().items():
            shadow = self.shadow[key]
            if torch.is_inference(shadow):
                # Be defensive if a validation-only path created or restored
                # this tensor under torch.inference_mode().
                shadow = shadow.detach().clone()
                self.shadow[key] = shadow
            if value.is_floating_point():
                shadow.mul_(self.decay).add_(
                    value.detach(), alpha=1 - self.decay
                )
            else:
                shadow.copy_(value.detach())

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            if key in self.shadow and self.shadow[key].shape == value.shape:
                reference = self.shadow[key]
                self.shadow[key] = value.detach().to(
                    device=reference.device, dtype=reference.dtype
                ).clone()
