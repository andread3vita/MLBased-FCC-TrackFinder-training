"""Shared checkpoint-resume fixes for the matched CIRCE/GATr trainers."""

from __future__ import annotations


def reset_validation_loop_progress(checkpoint):
    """Make the first validation after a resume run every validation batch.

    Validation checkpoints are written at validation end, so they store the
    validation loop as finished. Lightning restores that progress and seeds the
    restarted fetcher with the completed batch count, which stops the repeated
    validation after one batch per rank. Clearing the current progress here,
    before Lightning restores its loops, restarts validation from batch zero.
    Returns whether a stored validation progress was reset.
    """
    fit_loop = checkpoint.get("loops", {}).get("fit_loop", {})
    progress = fit_loop.get("epoch_loop.val_loop.batch_progress")
    if not progress or "current" not in progress:
        return False
    progress["current"] = {key: 0 for key in progress["current"]}
    progress["is_last_batch"] = False
    return True
