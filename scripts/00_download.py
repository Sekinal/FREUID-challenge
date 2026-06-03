#!/usr/bin/env python
"""Download & extract the FREUID competition data via the Kaggle API.

Idempotent: skips download if the zip already exists, and extraction if the
labels CSV is already present. Reads the standalone KGAT token from
``~/.kaggle/access_token`` (or the ``KAGGLE_API_TOKEN`` env var).

    uv run scripts/00_download.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config  # noqa: E402


def ensure_token() -> None:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    token_file = Path.home() / ".kaggle" / "access_token"
    if token_file.exists():
        os.environ["KAGGLE_API_TOKEN"] = token_file.read_text().strip()
        return
    sys.exit(
        "No Kaggle credentials found. Set KAGGLE_API_TOKEN or write the KGAT "
        "token to ~/.kaggle/access_token."
    )


def main() -> None:
    ensure_token()
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = config.RAW_DIR / f"{config.COMPETITION}.zip"
    if zip_path.exists():
        print(f"[skip] {zip_path.name} already downloaded ({zip_path.stat().st_size:,} B)")
    else:
        print(f"[download] {config.COMPETITION} -> {config.RAW_DIR}")
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", config.COMPETITION,
             "-p", str(config.RAW_DIR)],
            check=True,
        )

    if config.LABELS_CSV.exists():
        print(f"[skip] already extracted -> {config.EXTRACTED_DIR}")
    else:
        print(f"[extract] {zip_path.name} -> {config.EXTRACTED_DIR}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(config.EXTRACTED_DIR)

    n_images = len(list(config.EXTRACTED_DIR.rglob("*.jpeg")))
    print(f"[done] extracted {n_images} jpeg(s); labels: {config.LABELS_CSV.exists()}")


if __name__ == "__main__":
    main()
