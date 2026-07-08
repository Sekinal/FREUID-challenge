"""Schema-agnostic data loading & discovery helpers.

These functions tolerate the public-sample layout *and* a larger future release:
they discover image files by scanning, resolve image ids to real paths, and split
the composite ``type`` column into ``country`` / ``doc_type``.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config


# --------------------------------------------------------------------------
# Image discovery
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def image_index(images_dir: Optional[Path] = None) -> dict[str, Path]:
    """Map ``image_id`` (filename stem) -> absolute path, scanning recursively.

    Cached because every script needs it and the directory does not change
    within a run.
    """
    root = Path(images_dir) if images_dir else config.IMAGES_DIR
    index: dict[str, Path] = {}
    if not root.exists():
        return index
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS:
            index[p.stem] = p
    return index


def resolve_image_path(image_id: str) -> Optional[Path]:
    """Return the on-disk path for an image id, or ``None`` if missing."""
    return image_index().get(str(image_id))


def list_images() -> list[Path]:
    return sorted(image_index().values())


# --------------------------------------------------------------------------
# Tabular loaders
# --------------------------------------------------------------------------
def load_labels(resolve_paths: bool = True) -> pd.DataFrame:
    """Load the labels table, enriched with resolved paths and split ``type``.

    Adds columns: ``abs_path`` (str|NaN), ``path_exists`` (bool),
    ``country``, ``doc_type``.
    """
    df = pd.read_csv(config.LABELS_CSV)

    if config.TYPE_COL in df.columns:
        parts = df[config.TYPE_COL].astype(str).str.split("/", n=1, expand=True)
        df["country"] = parts[0].str.strip()
        df["doc_type"] = parts[1].str.strip() if parts.shape[1] > 1 else pd.NA

    if resolve_paths and config.ID_COL in df.columns:
        idx = image_index()
        df["abs_path"] = df[config.ID_COL].astype(str).map(
            lambda i: str(idx[i]) if i in idx else pd.NA
        )
        df["path_exists"] = df["abs_path"].notna()

    return df


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(config.SAMPLE_SUBMISSION_CSV)


def attach_image_paths(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``abs_path`` / ``path_exists`` columns by scanning ``IMAGES_DIR``."""
    out = df.copy()
    idx = image_index()
    out["abs_path"] = out[config.ID_COL].astype(str).map(
        lambda i: str(idx[i]) if i in idx else pd.NA
    )
    out["path_exists"] = out["abs_path"].notna()
    return out


def filter_with_images(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose image id resolves on disk."""
    out = attach_image_paths(df)
    return out[out["path_exists"]].reset_index(drop=True)


# --------------------------------------------------------------------------
# Artifact helpers
# --------------------------------------------------------------------------
def save_json(name: str, obj) -> Path:
    """Write a JSON artifact under ``artifacts/`` and return its path."""
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.ARTIFACTS_DIR / name
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    return path


def load_json(name: str):
    path = config.ARTIFACTS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _json_default(o):
    """Make numpy / pandas scalars JSON-serialisable."""
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)
