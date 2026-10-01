"""Shared plotting style for all CIRCE paper figures.

Goal: modern, "FCC-futuristic" but with the restraint of an academic physics/ML
paper (not a blog post). White background, a clearly visible but fine grid, thin
crisp lines, and one semantic colour map built on an FCC-style blue / deep-purple
/ teal / amber palette (no reds). Import configures matplotlib on import.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt  # noqa: F401  (re-exported convenience)
from matplotlib.colors import to_rgb
from matplotlib.ticker import MultipleLocator
from cycler import cycler


def yticks(ax, step=0.1):
    """Force major y-ticks at least every `step` (default 0.1) — finer gridlines."""
    ax.yaxis.set_major_locator(MultipleLocator(step))

# ---- FCC-style palette (deep, saturated, professional; no red) --------------
_BLUE   = "#2B57D6"   # standard config / match
_PURPLE = "#6A2FBF"   # keep-all config / adopted OP / strict  (the accent)
_TEAL   = "#109BA0"   # efficiency
_AMBER  = "#D98A1F"   # fake / highlight
_CYAN   = "#1FA8D6"
_INK    = "#141B2E"

COL = {
    "standard": _BLUE,
    "keepall":  _PURPLE,
    "best":     _PURPLE,
    "hilite":   _AMBER,     # contrast highlight over gradient families
    "grey":     "#AEB7C9",
    "ref":      "#6B7488",
    "noise":    "#C7CDD9",
    "match":    _BLUE,
    "strict":   _PURPLE,
    "eff":      _TEAL,
    "fake":     _AMBER,
    "cum":      _PURPLE,
    "ink":      _INK,
}
CYCLE = [_BLUE, _PURPLE, _TEAL, _AMBER, _CYAN]
TRACK_CMAP = "tab20"          # many-track id colouring (event, t-SNE)


def _seq_cmap():
    """Sequential colormap for gradient families (scientific, no red/purple clash)."""
    try:
        import seaborn as sns
        return sns.color_palette("mako", as_cmap=True)
    except Exception:
        return mpl.cm.get_cmap("cividis")


SEQ_CMAP = _seq_cmap()


def apply_style() -> None:
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    except Exception:
        pass
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.facecolor": "white",
        "axes.edgecolor": "#3C4657",  # dark, crisp axes
        "axes.linewidth": 1.1,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": _INK,
        "axes.labelsize": 11,
        "axes.labelcolor": "#26304A",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#C4CCDA",
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,
        "text.color": _INK,
        "xtick.color": "#3C4657",
        "ytick.color": "#3C4657",
        "xtick.labelcolor": "#3C4657",
        "ytick.labelcolor": "#3C4657",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "lines.markeredgewidth": 0.7,
        "lines.markeredgecolor": "white",
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#C9D0DE",
        "legend.facecolor": "white",
        "image.cmap": "cividis",
        "axes.prop_cycle": cycler(color=CYCLE),
    })


def gradient_fill(ax, x, y, color, alpha=0.16, y0=None, zorder=1):
    """Subtle vertical color->transparent fill under (x, y). Kept faint so it
    reads as a modern accent, not a decorative blog-post flourish."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    y0 = float(np.nanmin(y)) if y0 is None else y0
    rgb = to_rgb(color)
    grad = np.empty((256, 1, 4)); grad[:, 0, :3] = rgb
    grad[:, 0, 3] = np.linspace(alpha, 0.0, 256)
    im = ax.imshow(grad, extent=[x.min(), x.max(), y0, float(np.nanmax(y))],
                   origin="upper", aspect="auto", zorder=zorder)
    from matplotlib.patches import Polygon
    verts = np.column_stack([np.r_[x, x[::-1]], np.r_[y, np.full_like(y, y0)]])
    im.set_clip_path(Polygon(verts, closed=True, transform=ax.transData))
    return im


apply_style()
