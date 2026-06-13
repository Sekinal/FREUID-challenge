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
SPLITS_DIR = ARTIFACTS_DIR / "splits"     # train/val/test + CV fold manifests
RUNS_DIR = REPO_ROOT / "runs"           # checkpoints (git-ignored)
SUBMISSIONS_DIR = REPO_ROOT / "submissions"

# --- Kaggle competition ---------------------------------------------------
COMPETITION = "the-freuid-challenge-2026-ijcai-ecai"


def _pick_labels_csv() -> Path:
    for name in ("train_labels.csv", "train_sample_labels.csv"):
        path = EXTRACTED_DIR / name
        if path.exists():
            return path
    return EXTRACTED_DIR / "train_sample_labels.csv"


def _pick_images_dir() -> Path:
    candidates = (
        EXTRACTED_DIR / "train" / "train",
        EXTRACTED_DIR / "train_sample" / "train_sample",
    )
    best: Path | None = None
    best_count = -1
    for root in candidates:
        if not root.exists():
            continue
        count = sum(1 for _ in root.rglob("*.jpeg"))
        if count > best_count:
            best = root
            best_count = count
    return best or candidates[0]


# --- Known files / schema -------------------------------------------------
# Relative `image_path` in CSV may not match on-disk nesting; resolve by id.
LABELS_CSV = _pick_labels_csv()
SAMPLE_SUBMISSION_CSV = EXTRACTED_DIR / "sample_submission.csv"
IMAGES_DIR = _pick_images_dir()
PUBLIC_TEST_DIR = EXTRACTED_DIR / "public_test" / "public_test"

ID_COL = "id"
PATH_COL = "image_path"
LABEL_COL = "label"          # 0 = genuine/benign, 1 = fraud
IS_DIGITAL_COL = "is_digital"
TYPE_COL = "type"            # "<COUNTRY>/<DOC_TYPE>", e.g. "MAURITIUS/ID"

POSITIVE_LABEL = 1           # fraud is the positive class
LABEL_NAMES = {0: "genuine", 1: "fraud"}

IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff")

SPLIT_NAMES = ("train", "val", "test")


def is_full_dataset() -> bool:
    return LABELS_CSV.name == "train_labels.csv"


def ensure_dirs() -> None:
    for d in (ARTIFACTS_DIR, FIGURES_DIR, EMB_DIR, REPORT_DIR, SPLITS_DIR, RUNS_DIR, SUBMISSIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()
