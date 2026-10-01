"""Training/validation loss curve for the paper (fig:loss).

Reads the base run's epoch_metrics.csv (columns: epoch, mean_train_loss,
val_loss, val_match_loose, val_match_strict50, ...) and plots train vs val
object-condensation loss per epoch, with the strict-match rate on a twin axis
as a light reference. Shared paper style. Pure CPU/IO.
  python -m src.eval.plot_loss_curve --csv <run>/epoch_metrics.csv \
      --out /home/marko.cechovic/cgatr-paper/figures/loss_curve
"""
from __future__ import annotations
import argparse
import polars as pl
import matplotlib.pyplot as plt
from src.eval.plotstyle import COL, gradient_fill, yticks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="/home/marko.cechovic/cgatr-paper/figures/loss_curve")
    a = ap.parse_args()

    df = pl.read_csv(a.csv).sort("epoch")
    # Epochs Lightning did not validate carry no measurement -- see the note in train.py's
    # _EpochCSVCallback. Dropping them leaves a gap in the line, which is what an
    # unmeasured epoch should look like; plotting them would draw a flat segment that
    # never happened. Older CSVs, written before that was recorded as missing, contain the
    # duplicate instead, and this cannot distinguish those from a genuine plateau.
    n_before = len(df)
    df = df.filter(pl.col("val_loss").is_not_nan() & pl.col("val_loss").is_not_null())
    if len(df) < n_before:
        print(f"dropped {n_before - len(df)} epoch(s) with no validation")

    ep = df["epoch"].to_numpy()
    tr = df["mean_train_loss"].to_numpy()
    va = df["val_loss"].to_numpy()
    strict = df["val_match_strict50"].to_numpy() if "val_match_strict50" in df.columns else None

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    gradient_fill(ax, ep, tr, COL["standard"], alpha=0.28, y0=0.0)
    ax.plot(ep, tr, "o-", color=COL["standard"], ms=5, label="training loss", zorder=4)
    ax.plot(ep, va, "s--", color=COL["keepall"], ms=5, label="validation loss", zorder=5)
    ax.set_ylim(bottom=0); yticks(ax, 0.1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Object-condensation loss")
    ax.set_xlim(ep.min() - 0.5, ep.max() + 0.5)

    lines, labels = ax.get_legend_handles_labels()
    if strict is not None:
        ax2 = ax.twinx()
        ax2.grid(False)
        ax2.plot(ep, strict, "^:", color="0.5", ms=3.5, lw=1.2,
                 label="val. strict match")
        ax2.set_ylabel("Validation strict match", color="0.4")
        ax2.tick_params(axis="y", labelcolor="0.4")
        ax2.set_ylim(0, max(strict.max() * 1.25, 0.1))
        l2, lab2 = ax2.get_legend_handles_labels()
        lines += l2; labels += lab2
    ax.legend(lines, labels, loc="lower center", ncol=1)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(f"{a.out}.{e}")
    print("wrote", f"{a.out}.png")


if __name__ == "__main__":
    main()
