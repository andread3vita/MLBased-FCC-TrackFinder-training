"""Select raw or EMA model weights from a training checkpoint."""

from collections.abc import Mapping


WEIGHT_SOURCES = ("raw", "ema")


def select_checkpoint_weights(checkpoint, weights_source="ema"):
    """Return the explicitly requested model state from ``checkpoint``.

    EMA selection is intentionally strict: inference must not silently use raw
    weights when the requested EMA state is absent. Plain PyTorch state dicts
    remain supported through ``weights_source="raw"``.
    """
    if weights_source not in WEIGHT_SOURCES:
        raise ValueError(
            f"Unknown weights source {weights_source!r}; expected one of "
            f"{WEIGHT_SOURCES}"
        )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint is not a mapping")

    if weights_source == "ema":
        state = checkpoint.get("ema_state_dict")
        if not isinstance(state, Mapping) or not state:
            raise ValueError(
                "Checkpoint does not contain a non-empty ema_state_dict; "
                "use --weights-source raw for a legacy or raw-only checkpoint"
            )
    else:
        state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state, Mapping) or not state:
            raise ValueError("Checkpoint does not contain a non-empty raw state_dict")

    return state
