#!/usr/bin/env python
"""Inventory the extracted competition data.

Walks ``data/extracted/``: counts files by extension, total/per-type sizes,
introspects every CSV (columns, dtypes, row count, head), and characterises the
``sample_submission.csv`` format. Writes ``artifacts/inventory.json`` and a
file-type bar chart.

    uv run scripts/01_inventory.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, io, viz  # noqa: E402


def human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} TB"


def describe_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "n_unique": {c: int(df[c].nunique(dropna=True)) for c in df.columns},
        "head": df.head(5).to_dict(orient="records"),
    }


def main() -> None:
    viz.set_style()
    root = config.EXTRACTED_DIR
    if not root.exists():
        sys.exit("No extracted data. Run scripts/00_download.py first.")

    ext_counts: dict[str, int] = defaultdict(int)
    ext_bytes: dict[str, int] = defaultdict(int)
    total_files = 0
    total_bytes = 0
    csv_paths: list[Path] = []

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower() or "<none>"
        size = p.stat().st_size
        ext_counts[ext] += 1
        ext_bytes[ext] += size
        total_files += 1
        total_bytes += size
        if ext == ".csv":
            csv_paths.append(p)

    csvs = {p.name: describe_csv(p) for p in sorted(csv_paths)}

    inventory = {
        "extracted_root": str(root),
        "total_files": total_files,
        "total_bytes": total_bytes,
        "total_human": human(total_bytes),
        "by_extension": {
            ext: {"count": ext_counts[ext], "bytes": ext_bytes[ext],
                  "human": human(ext_bytes[ext])}
            for ext in sorted(ext_counts, key=lambda e: -ext_bytes[e])
        },
        "n_images_indexed": len(io.image_index()),
        "csvs": csvs,
    }
    out = io.save_json("inventory.json", inventory)

    # --- figure: file count by extension ---
    exts = list(inventory["by_extension"].keys())
    counts = [inventory["by_extension"][e]["count"] for e in exts]
    fig, ax = viz.plt.subplots(figsize=(6, 3.5))
    ax.bar(exts, counts, color=viz.ACCENT)
    viz.annotate_bars(ax)
    ax.set(title="Files by extension", xlabel="extension", ylabel="count")
    viz.save_fig(fig, "inventory_file_types.png")

    # --- console summary ---
    print(f"Extracted root : {root}")
    print(f"Total files    : {total_files}  ({human(total_bytes)})")
    for ext, meta in inventory["by_extension"].items():
        print(f"  {ext:8s} {meta['count']:>6} files  {meta['human']:>10}")
    print(f"Images indexed : {inventory['n_images_indexed']}")
    for name, meta in csvs.items():
        print(f"\nCSV {name}: {meta['rows']} rows, columns={meta['columns']}")
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
