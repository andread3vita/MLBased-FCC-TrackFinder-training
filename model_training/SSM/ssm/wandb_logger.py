"""Independent Weights & Biases logging for SSM (offline-mode friendly).

No code shared with other arms. All functions are no-ops when disabled.
On the offline GPU machine use WANDB_MODE=offline (set by slurmTrain.sh) and
sync later from a connected machine:  wandb sync <run_dir>/wandb/offline-run-*
"""

from __future__ import annotations

from pathlib import Path

_run = None


_CANONICAL_HPARAMS = {
    "epochs": "epochs", "optimizer": None, "learning_rate_schedule": None,
    "start_learning_rate": "lr", "weight_decay": "weight_decay",
    "learning_rate_warmup_epochs": "warmup_epochs",
    "gradient_clip_value": "clip", "embedding_dimension": "embedding_dim",
    "number_of_blocks": "n_layer", "qmin": "qmin",
    "attractive_weight": "attr_weight", "repulsive_weight": "repul_weight",
    "beta_suppress_weight": "beta_suppress_weight",
    "embedding_variance_weight": "var_weight",
    "embedding_variance_warmup_epochs": "var_warmup_epochs",
    "sweep_tbeta_grid": "sweep_tbeta_grid", "sweep_td_grid": "sweep_td_grid",
    "sweep_max_events": "sweep_events", "seed": "seed",
}


def start(config: dict, out_dir: str, name: str | None,
          project: str | None, entity: str | None, enabled: bool):
    global _run
    if not enabled:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed in this env — continuing without logging "
              "(install wheel or run with --no-log-wandb)", flush=True)
        return None
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    canon = {
        "comparison_contract": "ssm_v1_backbone_only",
        "model_family": "SSM_MAMBA2_SSD",
        "loss_function": "ssm.oc_loss.object_condensation_loss (same math as shared circe_loss)",
    }
    for canonical, src in _CANONICAL_HPARAMS.items():
        canon[canonical] = config.get(src) if src else (
            "adamw" if canonical == "optimizer" else "warmup+plateau")
    config = {**config, **canon}
    _run = wandb.init(
        project=project or None,
        entity=entity or None,
        name=name or None,
        dir=out_dir,
        config=config,
    )
    print(f"wandb: logging run '{_run.name}' (mode={wandb.run.settings.mode}) -> {out_dir}/wandb",
          flush=True)
    return _run


def enabled() -> bool:
    return _run is not None


def log(metrics: dict, step: int):
    if _run is not None:
        _run.log(metrics, step=step)


def log_media(media: dict, step: int):
    """media: {wandb_key: png_path} — logged as images."""
    if _run is None or not media:
        return
    import wandb
    _run.log({k: wandb.Image(str(v)) for k, v in media.items()}, step=step)


def finish():
    global _run
    if _run is not None:
        _run.finish()
        _run = None
