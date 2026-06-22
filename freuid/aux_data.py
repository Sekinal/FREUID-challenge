"""Auxiliary external datasets for pre-mixing into FREUID training.

Currently supports IDNet-2025 (cactuslab/IDNet-2025, CC-BY-4.0): per-country
folders with fraud-labeled identity documents. Used to enrich the FREUID train
split with extra fraud / genuine examples. Validation/test stay FREUID-only.

Label mapping (per the dataset card):
    positive/                    -> 0  (genuine / bona-fide)
    fraud5_inpaint_and_rewrite/  -> 1  (fraud)
    fraud6_crop_and_replace/     -> 1  (fraud)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config

# folder name -> label
IDNET_LABEL_DIRS = {
    "positive": 0,
    "fraud5_inpaint_and_rewrite": 1,
    "fraud6_crop_and_replace": 1,
}

_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _scan_root(root: Path) -> list[dict]:
    tag = root.name  # e.g. "idnet2025" or "idnet2025_scanned" -> keeps ids unique
    is_digital = 0 if "scanned" in tag else 1
    rows: list[dict] = []
    if not root.exists():
        return rows
    for country_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        country = country_dir.name
        for sub, label in IDNET_LABEL_DIRS.items():
            leaf = country_dir / sub
            if not leaf.is_dir():
                continue
            for img in leaf.iterdir():
                if img.suffix.lower() not in _IMG_EXT:
                    continue
                rows.append(
                    {
                        config.ID_COL: f"idnet_{tag}_{country}_{sub}_{img.stem}",
                        config.LABEL_COL: label,
                        "abs_path": str(img),
                        config.TYPE_COL: f"IDNET/{tag}/{country}",
                        "country": country,
                        config.IS_DIGITAL_COL: is_digital,
                    }
                )
    return rows


def load_idnet_frame(roots: Path | str | list) -> pd.DataFrame:
    """Scan one or more IDNet-2025 extractions into a FREUID-compatible frame.

    ``roots`` may be a single path or a list of paths (e.g. digital + scanned).
    Columns: ``id``, ``label``, ``abs_path``, ``type``, ``country``,
    ``is_digital`` (0 for *_scanned roots = print-and-capture, else 1).
    Each country lives in ``<root>/<COUNTRY>/<label_dir>/*.jpg``.
    """
    if isinstance(roots, (str, Path)):
        roots = [roots]
    rows: list[dict] = []
    for r in roots:
        rows.extend(_scan_root(Path(r)))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No IDNet images found under {roots}")
    return frame
