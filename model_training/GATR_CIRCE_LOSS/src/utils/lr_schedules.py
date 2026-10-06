import math


REDUCE_ON_PLATEAU_UNIT = "reduceplateau-v1"
EPOCH_WARMUP_COSINE_UNIT = "epoch-v1"
STEP_WARMUP_COSINE_UNIT = "step-v2"
NO_SCHEDULER_UNIT = "none-v1"


def epoch_warmup_cosine_factor(
    epoch: int,
    total_epochs: int,
    warmup_epochs: float,
    min_ratio: float,
) -> float:
    """Return an epoch-based warmup plus cosine-decay LR multiplier."""
    total_epochs = max(int(total_epochs), 1)
    epoch = max(float(epoch), 0.0)
    min_ratio = min(max(float(min_ratio), 0.0), 1.0)

    if total_epochs == 1:
        return 1.0

    # Reserve at least the final epoch for the end of the cosine schedule.
    warmup_epochs = min(
        max(float(warmup_epochs), 0.0), float(total_epochs - 1)
    )
    if warmup_epochs > 0.0 and epoch < warmup_epochs:
        return max(min_ratio, epoch / warmup_epochs)

    decay_span = max(float(total_epochs - 1) - warmup_epochs, 1.0)
    progress = min(max((epoch - warmup_epochs) / decay_span, 0.0), 1.0)
    return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))


def migrate_step_scheduler_checkpoint(
    checkpoint: dict,
    resume_epoch: int,
    start_lr: float,
    factor: float,
) -> None:
    """Convert optimizer/LambdaLR state from step units to epoch units in place."""
    optimizer_base_lrs = []
    for optimizer_state in checkpoint.get("optimizer_states", []):
        base_lrs = []
        for group in optimizer_state.get("param_groups", []):
            base_lr = float(group.get("initial_lr", start_lr))
            group["initial_lr"] = base_lr
            group["lr"] = base_lr * factor
            base_lrs.append(base_lr)
        if base_lrs:
            optimizer_base_lrs.append(base_lrs)

    for index, scheduler_state in enumerate(checkpoint.get("lr_schedulers", [])):
        base_lrs = (
            optimizer_base_lrs[index]
            if index < len(optimizer_base_lrs)
            else [float(start_lr)]
        )
        scheduler_state["base_lrs"] = base_lrs
        scheduler_state["last_epoch"] = int(resume_epoch)
        scheduler_state["_step_count"] = int(resume_epoch) + 1
        scheduler_state["_last_lr"] = [lr * factor for lr in base_lrs]


def checkpoint_resume_metadata(checkpoint: dict) -> dict:
    """Return the optimizer LR and scheduler that a full resume must restore."""
    optimizer_states = checkpoint.get("optimizer_states") or []
    learning_rates = []
    for optimizer_state in optimizer_states:
        learning_rates.extend(
            float(group["lr"])
            for group in optimizer_state.get("param_groups", [])
            if group.get("lr") is not None
        )

    hyperparameters = checkpoint.get("hyper_parameters") or {}
    scheduler_name = hyperparameters.get("lr_scheduler")
    if scheduler_name is not None:
        scheduler_name = str(scheduler_name).lower()

    schedule_unit = checkpoint.get("lr_schedule_unit")
    if schedule_unit == REDUCE_ON_PLATEAU_UNIT:
        scheduler_name = "reduceplateau"
    elif schedule_unit == NO_SCHEDULER_UNIT:
        scheduler_name = "none"
    elif (
        schedule_unit in (EPOCH_WARMUP_COSINE_UNIT, STEP_WARMUP_COSINE_UNIT)
        and scheduler_name != "none"
    ):
        # Legacy non-plateau names all used this same epoch LambdaLR. Restore
        # them through the single canonical name that remains supported.
        scheduler_name = "flat+decay"

    return {
        "contains_training_state": bool(optimizer_states),
        "learning_rates": learning_rates,
        "lr_scheduler": scheduler_name,
        "lr_schedule_unit": schedule_unit,
    }
