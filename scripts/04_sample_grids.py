#!/usr/bin/env python
"""Visual montages of sample documents.

Builds an overall thumbnail grid, a genuine-vs-fraud grid, and a per-document-
type grid so a human can eyeball what the data actually looks like. Saves PNGs
into ``figures/``.

    uv run scripts/04_sample_grids.py
"""
from __future__ import annotations

import math
import sys
import textwrap
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, io, viz  # noqa: E402

THUMB = 256  # max thumbnail edge


def _thumb(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail((THUMB, THUMB))
    return im


def grid(records, name: str, title: str, ncols: int = 5) -> None:
    """records: list of (path, caption). Renders a labelled grid figure."""
    records = [r for r in records if r[0] is not None]
    if not records:
        return
    ncols = min(ncols, len(records))
    nrows = math.ceil(len(records) / ncols)
    fig, axes = viz.plt.subplots(nrows, ncols, figsize=(ncols * 2.4, nrows * 2.7))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax in axes:
        ax.axis("off")
    for ax, (path, caption) in zip(axes, records):
        try:
            ax.imshow(_thumb(path))
        except Exception as exc:  # noqa: BLE001
            ax.text(.5, .5, f"unreadable\n{exc}", ha="center", va="center", fontsize=6)
        ax.set_title("\n".join(textwrap.wrap(caption, 22)), fontsize=7)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    viz.save_fig(fig, name)


def main() -> None:
    viz.set_style()
    df = io.load_labels()
    df = df[df["path_exists"]] if "path_exists" in df else df
    if df.empty:
        sys.exit("No resolvable images. Run scripts/00_download.py first.")

    def cap(row) -> str:
        lab = config.LABEL_NAMES.get(row[config.LABEL_COL], row[config.LABEL_COL])
        dig = "digital" if row.get(config.IS_DIGITAL_COL) else "physical"
        return f"{lab} | {row.get('type', '?')} | {dig}"

    # overall grid
    grid([(Path(r["abs_path"]), cap(r)) for _, r in df.iterrows()],
         "grid_overall.png", "Sample documents (all)")

    # genuine vs fraud (balanced)
    for lab_val, lab_name in config.LABEL_NAMES.items():
        sub = df[df[config.LABEL_COL] == lab_val]
        grid([(Path(r["abs_path"]), cap(r)) for _, r in sub.iterrows()],
             f"grid_label_{lab_name}.png", f"Label = {lab_name} (n={len(sub)})")

    # per doc_type
    if "doc_type" in df.columns:
        for dt, sub in df.groupby("doc_type"):
            safe = str(dt).replace("/", "_")
            grid([(Path(r["abs_path"]), cap(r)) for _, r in sub.iterrows()],
                 f"grid_doctype_{safe}.png", f"doc_type = {dt} (n={len(sub)})")

    print(f"[done] grids written to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
