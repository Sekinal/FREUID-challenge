#!/usr/bin/env bash
# Run the full EDA pipeline end-to-end. Intended to be run from the repo root:
#   bash scripts/run_eda.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="$HOME/.local/bin:$PATH"

run() { echo -e "\n=== $1 ==="; uv run "python" "$1"; }

uv run python scripts/00_download.py
run scripts/01_inventory.py
run scripts/02_labels.py
run scripts/03_image_stats.py
run scripts/04_sample_grids.py
run scripts/05_duplicates_leakage.py
run scripts/06_embeddings_gpu.py
run scripts/07_clustering_umap.py
run scripts/08_build_report.py

echo -e "\nAll done. Report: report/report.pdf"
