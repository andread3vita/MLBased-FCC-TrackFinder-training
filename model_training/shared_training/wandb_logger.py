"""One Lightning/W&B logging implementation for the matched trainers."""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch.loggers import WandbLogger

from shared_training.logging_contract import allowed_wandb_metric


class MatchedWandbLogger(WandbLogger):
    """W&B logger that exposes only the shared comparison metrics."""

    _COMMON_HPARAMS = {
        "batch_size": ("batch_size",),
        "epochs": ("num_epochs",),
        "optimizer": ("optimizer",),
        "learning_rate_schedule": ("lr_schedule", "lr_scheduler"),
        "start_learning_rate": ("start_lr",),
        "minimum_learning_rate": ("min_lr",),
        "weight_decay": ("weight_decay",),
        "learning_rate_warmup_epochs": ("warmup_epochs",),
        "gradient_clip_value": ("gradient_clip_val",),
        "ema_decay": ("ema_decay",),
        "embedding_dimension": ("embed_dim", "clustering_space_dim"),
        "number_of_blocks": ("num_blocks", "gatr_blocks"),
        "hidden_multivector_channels": ("hidden_mv_channels",),
        "hidden_scalar_channels": ("hidden_s_channels",),
        "qmin": ("qmin",),
        "attractive_weight": ("attr_weight", "L_attractive_weight"),
        "repulsive_weight": ("repul_weight", "L_repulsive_weight"),
        "beta_suppress_weight": ("beta_suppress_weight",),
        "embedding_variance_weight": ("var_weight",),
        "embedding_variance_warmup_epochs": ("var_warmup_epochs",),
        "sweep_tbeta_grid": ("sweep_tbeta_grid",),
        "sweep_td_grid": ("sweep_td_grid",),
        "sweep_min_hits_grid": ("sweep_min_hits_grid",),
        "sweep_max_events": ("validation_sweep_max_events",),
        "sweep_match_metric": ("sweep_match_metric",),
        "sweep_truth_min_hits": ("sweep_truth_min_hits",),
        "rejected_seed_policy": ("rejected_seed_policy",),
        "seed": ("seed",),
    }

    def log_hyperparams(self, params):
        """Upload one canonical config schema for both model families."""
        source = vars(params) if hasattr(params, "__dict__") else dict(params)
        model_family = (
            "CIRCE" if "embed_dim" in source else "GATR_CIRCE_LOSS"
        )
        common = {
            "comparison_contract": "circe_gatr_shared_v1",
            "model_family": model_family,
            "loss_function": "shared_training.circe_loss.object_condensation_loss",
            "tracking_metrics": "current_GATR_tracking_metrics",
        }
        for canonical_name, aliases in self._COMMON_HPARAMS.items():
            value = next(
                (source[name] for name in aliases if name in source),
                None,
            )
            if canonical_name == "optimizer" and isinstance(value, str):
                value = value.lower()
            common[canonical_name] = value
        super().log_hyperparams(common)

    def log_metrics(self, metrics, step=None):
        common = {
            key: value for key, value in metrics.items()
            if allowed_wandb_metric(key)
        }
        if common:
            super().log_metrics(common, step=step)


def build_experiment_logger(
    *, enabled, output_dir, project=None, entity=None, run_name=None,
    csv_name="lightning_logs",
):
    """Build the common W&B logger, or a local CSV logger when disabled."""
    if enabled:
        try:
            import wandb  # noqa: F401 - fail here with a useful message
        except ImportError as error:
            raise RuntimeError(
                "W&B logging was requested but the wandb package is unavailable"
            ) from error
        return MatchedWandbLogger(
            project=project,
            entity=entity,
            name=run_name,
            save_dir=str(output_dir),
        )

    from lightning.pytorch.loggers import CSVLogger
    return CSVLogger(str(output_dir), name=csv_name)


def wandb_image(path):
    """Create a W&B image lazily, so non-W&B runs need no wandb import."""
    import wandb
    return wandb.Image(str(path))


def wandb_html(payload):
    """Create a W&B HTML object lazily."""
    import wandb
    return wandb.Html(payload)


def log_wandb_media(logger, media):
    """Log only the media common to both comparison arms."""
    common_prefixes = (
        "plots/operating_point_sweep/",
        "plots/tracking_efficiency_vs_pt/",
        "plots/tracking_efficiency_vs_displacement/",
    )
    common_keys = {
        "plots/validation_event_0/hits_by_mc_particle",
        "plots/validation_event_0/hits_by_reconstructed_particle",
    }
    common = {
        key: value for key, value in media.items()
        if key in common_keys or key.startswith(common_prefixes)
    }
    if common and hasattr(logger, "experiment"):
        logger.experiment.log(common)


def operating_point_media(output_dir):
    """Build the shared W&B image payload produced by the common plotter."""
    output = Path(output_dir)
    paths = [output / "efficiency_fake_pareto.png"]
    paths.extend(sorted(output.glob("sweep_min_hits_*.png")))
    return {
        f"plots/operating_point_sweep/{path.stem}": wandb_image(path)
        for path in paths if path.is_file()
    }


def tracking_efficiency_media(pt_paths, displacement_paths):
    """Build the identical pT/displacement image payload for both models."""
    media = {}
    for name, path in pt_paths.items():
        if path is not None and Path(path).is_file():
            media[f"plots/tracking_efficiency_vs_pt/{name}"] = wandb_image(path)
    for name, path in displacement_paths.items():
        if path is not None and Path(path).is_file():
            media[
                f"plots/tracking_efficiency_vs_displacement/{name}"
            ] = wandb_image(path)
    return media
