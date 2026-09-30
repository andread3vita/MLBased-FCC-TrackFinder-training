def weighted_loss_component_metrics(
    components,
    *,
    attractive_weight: float,
    repulsive_weight: float,
    variance_weight: float,
):
    """Build W&B metrics containing loss contributions after weighting."""
    metrics = {}
    if len(components) >= 8:
        metrics.update(
            {
                "L_potential=w_attractive*L_attractive+w_repulsive*L_repulsive": components[0],
                "L_beta=L_beta_signal+L_beta_noise+w_suppress*L_beta_suppress+w_second*L_beta_second+w_variance*L_embedding_variance": components[1],
                "w_attractive*L_attractive": attractive_weight * components[2],
                "w_repulsive*L_repulsive": repulsive_weight * components[3],
                # These terms have fixed unit coefficients.
                "L_beta_signal": components[4],
                "L_beta_noise": components[5],
                # The suppression weight is already applied inside the loss.
                "w_suppress*L_beta_suppress": components[6],
                "w_variance*L_embedding_variance": variance_weight
                * components[7],
            }
        )
    if len(components) >= 10:
        # This returned component is already multiplied by beta_second_weight.
        metrics["w_second*L_beta_second"] = components[9]
    return metrics
