def resolve_validation_batch_limit(limit: int):
    """Translate the CLI convention ``-1 == all batches`` for Lightning."""
    limit = int(limit)
    if limit == -1:
        return 1.0
    if limit < 1:
        raise ValueError("validation batch limit must be -1 or a positive integer")
    return limit


def validation_output_name(validation_tag, epoch: int) -> str:
    """Name pre-training and epoch-end validation outputs unambiguously."""
    if validation_tag is not None:
        return str(validation_tag)
    return f"epoch_{int(epoch):04d}"
