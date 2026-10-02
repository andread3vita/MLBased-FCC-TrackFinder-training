"""Single metric contract for matched CIRCE/GATr W&B comparisons."""

from __future__ import annotations


TRAIN_METRIC_NAMES = {
    "loss": "train/loss",
    "attractive": "train/loss_attractive_weighted",
    "repulsive": "train/loss_repulsive_weighted",
    "beta_signal": "train/loss_beta_signal",
    "beta_noise": "train/loss_beta_noise",
    "beta_suppress": "train/loss_beta_suppress_weighted",
    "variance": "train/loss_embedding_variance_weighted",
    "variance_weight": "train/embedding_variance_weight",
    "learning_rate": "train/learning_rate",
}

VALIDATION_LOSS_NAME = "validation/loss"

SWEEP_METRIC_NAMES = {
    "pareto_f1": "validation/pareto_f1",
    "pareto_fake_rate": "validation/fake_rate_at_pareto_f1",
    "pareto_efficiency": "validation/tracking_efficiency_at_pareto_f1",
    "pareto_tbeta": "validation/pareto_f1_tbeta",
    "pareto_td": "validation/pareto_f1_td",
    "max_efficiency": "validation/max_tracking_efficiency",
    "max_efficiency_fake_rate": "validation/fake_rate_at_max_efficiency",
    "max_efficiency_f1": "validation/f1_at_max_efficiency",
    "max_efficiency_tbeta": "validation/max_efficiency_tbeta",
    "max_efficiency_td": "validation/max_efficiency_td",
}


def training_metric_values(
    loss, components, attractive_weight, repulsive_weight, variance_weight,
):
    """Return the exact common set of per-batch training values."""
    return {
        TRAIN_METRIC_NAMES["loss"]: loss,
        TRAIN_METRIC_NAMES["attractive"]: (
            float(attractive_weight) * components["L_V_att"]
        ),
        TRAIN_METRIC_NAMES["repulsive"]: (
            float(repulsive_weight) * components["L_V_rep"]
        ),
        TRAIN_METRIC_NAMES["beta_signal"]: components["L_beta_sig"],
        TRAIN_METRIC_NAMES["beta_noise"]: components["L_beta_noise"],
        TRAIN_METRIC_NAMES["beta_suppress"]: components["L_beta_suppress"],
        TRAIN_METRIC_NAMES["variance"]: (
            float(variance_weight) * components["L_var"]
        ),
        TRAIN_METRIC_NAMES["variance_weight"]: float(variance_weight),
    }


def log_training_metrics(
    module, loss, components, attractive_weight, repulsive_weight,
    variance_weight, batch_size,
):
    """Log identical training keys and reductions from either model."""
    values = training_metric_values(
        loss,
        components,
        attractive_weight,
        repulsive_weight,
        variance_weight,
    )
    for name, value in values.items():
        module.log(
            name,
            value,
            on_step=True,
            on_epoch=True,
            prog_bar=name == TRAIN_METRIC_NAMES["loss"],
            sync_dist=True,
            batch_size=batch_size,
        )
    module.log(
        TRAIN_METRIC_NAMES["learning_rate"],
        float(module.trainer.optimizers[0].param_groups[0]["lr"]),
        on_step=True,
        on_epoch=False,
        sync_dist=False,
        batch_size=batch_size,
    )
    return values


def log_validation_loss(module, loss, batch_size):
    """Log the shared validation-loss key."""
    module.log(
        VALIDATION_LOSS_NAME,
        loss,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        sync_dist=True,
        batch_size=batch_size,
    )


def sweep_metric_values(working_points):
    """Return the common scalar view of GATr's two selected working points."""
    pareto = working_points["pareto_f1"]
    maximum = working_points["max_efficiency"]
    return {
        SWEEP_METRIC_NAMES["pareto_f1"]: pareto["f1"],
        SWEEP_METRIC_NAMES["pareto_fake_rate"]: pareto["fake_rate"],
        SWEEP_METRIC_NAMES["pareto_efficiency"]: pareto["efficiency"],
        SWEEP_METRIC_NAMES["pareto_tbeta"]: pareto["tbeta"],
        SWEEP_METRIC_NAMES["pareto_td"]: pareto["td"],
        SWEEP_METRIC_NAMES["max_efficiency"]: maximum["efficiency"],
        SWEEP_METRIC_NAMES["max_efficiency_fake_rate"]: maximum["fake_rate"],
        SWEEP_METRIC_NAMES["max_efficiency_f1"]: maximum["f1"],
        SWEEP_METRIC_NAMES["max_efficiency_tbeta"]: maximum["tbeta"],
        SWEEP_METRIC_NAMES["max_efficiency_td"]: maximum["td"],
    }


def log_sweep_metrics(module, working_points):
    """Log identical operating-point scalar keys from either model."""
    for name, value in sweep_metric_values(working_points).items():
        module.log(
            name,
            float(value),
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )


def allowed_wandb_metric(name):
    """Whether a scalar key belongs to the matched comparison contract."""
    if name in ("epoch", "trainer/global_step"):
        return True
    bases = set(TRAIN_METRIC_NAMES.values())
    if name in bases or name in {VALIDATION_LOSS_NAME, *SWEEP_METRIC_NAMES.values()}:
        return True
    return any(name == f"{base}_{suffix}" for base in bases for suffix in ("step", "epoch"))

