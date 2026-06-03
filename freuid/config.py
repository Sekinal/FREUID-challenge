"""Central configuration: paths, competition constants, schema column names.

Everything is derived from the repo root so the scripts work identically on the
remote box and any mirror. Directories for generated artifacts are created on import.
"""
from __future__ import annotations

from pathlib import Path

# freuid/config.py -> repo root is one level up from the package dir.
REPO_ROOT = Path(__file__).resolve().parents[1]

# --- Data (git-ignored) ---------------------------------------------------
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTRACTED_DIR = DATA_DIR / "extracted"

# --- Generated outputs ----------------------------------------------------
ARTIFACTS_DIR = REPO_ROOT / "artifacts"   # small JSON/parquet stats (committed)
FIGURES_DIR = REPO_ROOT / "figures"       # PNGs embedded by the report (committed)
EMB_DIR = REPO_ROOT / "embeddings"        # large *.npy embeddings (git-ignored)
REPORT_DIR = REPO_ROOT / "report"         # Typst source + compiled PDF

# --- Kaggle competition ---------------------------------------------------
COMPETITION = "the-freuid-challenge-2026-ijcai-ecai"

# --- Known files / schema (from the public sample) ------------------------
# The labels CSV ships a relative `image_path` that does not match the real
# on-disk nesting (train_sample/train_sample/<id>.jpeg), so we resolve by id.
LABELS_CSV = EXTRACTED_DIR / "train_sample_labels.csv"
SAMPLE_SUBMISSION_CSV = EXTRACTED_DIR / "sample_submission.csv"
IMAGES_DIR = EXTRACTED_DIR / "train_sample" / "train_sample"

ID_COL = "id"
PATH_COL = "image_path"
LABEL_COL = "label"          # 0 = genuine/benign, 1 = fraud
IS_DIGITAL_COL = "is_digital"
TYPE_COL = "type"            # "<COUNTRY>/<DOC_TYPE>", e.g. "MAURITIUS/ID"

POSITIVE_LABEL = 1           # fraud is the positive class
LABEL_NAMES = {0: "genuine", 1: "fraud"}

IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def ensure_dirs() -> None:
    for d in (ARTIFACTS_DIR, FIGURES_DIR, EMB_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()
