#!/usr/bin/env python
"""Per-image property profiling + corruption check.

For every image: width, height, aspect ratio, megapixels, channels/mode,
format, on-disk bytes, and (best effort) JPEG DPI. Each file is fully decoded
to detect truncation/corruption. Writes ``artifacts/image_stats.parquet``,
``artifacts/image_stats_summary.json`` and distribution figures.

    uv run scripts/03_image_stats.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, io, viz  # noqa: E402

# We *want* to detect truncated files, so do NOT allow loading them silently.
ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = None  # disable decompression-bomb warning for big scans


def probe(path: Path) -> dict:
    rec: dict = {"abs_path": str(path), "id": path.stem,
                 "bytes": path.stat().st_size, "corrupt": False, "error": None}
    try:
        with Image.open(path) as im:
            rec["format"] = im.format
            rec["mode"] = im.mode
            rec["width"], rec["height"] = im.size
            dpi = im.info.get("dpi")
            if dpi:
                rec["dpi_x"], rec["dpi_y"] = float(dpi[0]), float(dpi[1])
            im.load()  # force full decode -> raises on truncation
            rec["channels"] = len(im.getbands())
    except Exception as exc:  # noqa: BLE001
        rec["corrupt"] = True
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec

    w, h = rec["width"], rec["height"]
    rec["megapixels"] = round(w * h / 1e6, 3)
    rec["aspect"] = round(w / h, 4) if h else None
    rec["orientation"] = "landscape" if w > h else ("portrait" if h > w else "square")
    rec["bytes_per_pixel"] = round(rec["bytes"] / (w * h), 4) if w * h else None
    return rec


def main() -> None:
    viz.set_style()
    labels = io.load_labels()
    label_map = dict(zip(labels[config.ID_COL].astype(str), labels[config.LABEL_COL]))

    paths = io.list_images()
    if not paths:
        sys.exit("No images found. Run scripts/00_download.py first.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = [probe(p) for p in paths]
    df = pd.DataFrame(rows)
    df["label"] = df["id"].map(label_map)
    df["label_name"] = df["label"].map(config.LABEL_NAMES)

    df.to_parquet(config.ARTIFACTS_DIR / "image_stats.parquet", index=False)

    ok = df[~df["corrupt"]]
    summary = {
        "n_images": int(len(df)),
        "n_corrupt": int(df["corrupt"].sum()),
        "corrupt_files": df.loc[df["corrupt"], ["id", "error"]].to_dict("records"),
        "formats": df["format"].value_counts(dropna=False).to_dict(),
        "modes": df["mode"].value_counts(dropna=False).to_dict(),
        "orientations": ok["orientation"].value_counts().to_dict() if len(ok) else {},
    }
    for col in ("width", "height", "aspect", "megapixels", "bytes", "bytes_per_pixel"):
        if col in ok and len(ok):
            s = ok[col].astype(float)
            summary[col] = {
                "min": float(s.min()), "p25": float(s.quantile(.25)),
                "median": float(s.median()), "p75": float(s.quantile(.75)),
                "max": float(s.max()), "mean": float(s.mean()),
            }
    out = io.save_json("image_stats_summary.json", summary)

    if len(ok):
        # distributions of width/height/aspect/filesize
        fig, axes = viz.plt.subplots(2, 2, figsize=(9, 6.5))
        specs = [("width", "Width (px)"), ("height", "Height (px)"),
                 ("aspect", "Aspect ratio (w/h)"), ("bytes", "File size (bytes)")]
        for ax, (col, title) in zip(axes.ravel(), specs):
            bins = min(20, max(5, len(ok)))
            ax.hist(ok[col].astype(float), bins=bins, color=viz.ACCENT, alpha=.85)
            ax.set(title=title, ylabel="count")
        viz.save_fig(fig, "image_property_hists.png")

        # width vs height scatter, colored by label
        fig, ax = viz.plt.subplots(figsize=(5.5, 5))
        for name, sub in ok.groupby("label_name"):
            ax.scatter(sub["width"], sub["height"], s=40, alpha=.8,
                       label=name, color=viz.LABEL_PALETTE.get(name, viz.ACCENT))
        ax.set(title="Resolution (width vs height)", xlabel="width", ylabel="height")
        ax.legend(title="label")
        viz.save_fig(fig, "image_resolution_scatter.png")

    print(f"images={len(df)}  corrupt={summary['n_corrupt']}  formats={summary['formats']}")
    if "width" in summary:
        print(f"width  median={summary['width']['median']:.0f} "
              f"range=[{summary['width']['min']:.0f},{summary['width']['max']:.0f}]")
        print(f"height median={summary['height']['median']:.0f} "
              f"range=[{summary['height']['min']:.0f},{summary['height']['max']:.0f}]")
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
