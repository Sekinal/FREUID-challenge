"""Consistent matplotlib styling and small plotting helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless box, no display
import matplotlib.pyplot as plt
import seaborn as sns

from . import config

# Color used consistently for the fraud/genuine split across the report.
LABEL_PALETTE = {"genuine": "#2a9d8f", "fraud": "#e76f51", 0: "#2a9d8f", 1: "#e76f51"}
ACCENT = "#264653"


def set_style() -> None:
    sns.set_theme(context="notebook", style="whitegrid", palette="deep")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "figure.autolayout": True,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "font.size": 10,
        }
    )


def save_fig(fig, name: str, *, tight: bool = True) -> Path:
    """Save a figure into ``figures/`` and close it. Returns the path."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    if tight:
        fig.savefig(path, bbox_inches="tight")
    else:
        fig.savefig(path)
    plt.close(fig)
    return path


def annotate_bars(ax, fmt: str = "{:.0f}") -> None:
    """Write the value on top of each bar in a bar chart."""
    for p in ax.patches:
        h = p.get_height()
        if h and h == h:  # not NaN
            ax.annotate(
                fmt.format(h),
                (p.get_x() + p.get_width() / 2, h),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 1),
                textcoords="offset points",
            )
