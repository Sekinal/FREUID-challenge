#!/usr/bin/env python
"""Build group-aware train / val / test splits and CV fold manifests.

Split policy:
  - Hold out whole document ``type`` groups (country/doc-type), never random rows.
  - Merge types linked by near-duplicate pairs (``artifacts/duplicates.json``).
  - Greedy stratified assignment by size + fraud rate.
  - GroupKFold CV on train+val using the same type components.

Outputs under ``artifacts/splits/``:
  - labeled_with_split.csv, train.csv, val.csv, test.csv
  - manifest.json, cv_fold_*.json

    uv run scripts/09_build_splits.py
    uv run scripts/09_build_splits.py --val-frac 0.15 --test-frac 0.15 --folds 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, io, splits, validation  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FREUID train/val/test splits.")
    p.add_argument("--val-frac", type=float, default=0.15, help="Fraction of type components for val.")
    p.add_argument("--test-frac", type=float, default=0.15, help="Fraction of type components for test.")
    p.add_argument("--folds", type=int, default=5, help="GroupKFold folds on train+val.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = splits.SplitConfig(
        val_type_fraction=args.val_frac,
        test_type_fraction=args.test_frac,
        n_cv_folds=args.folds,
        random_state=args.seed,
    )

    labels = io.load_labels()
    if labels.empty:
        sys.exit("No labels found. Run scripts/00_download.py first.")

    print(f"[info] labels: {config.LABELS_CSV.name} ({len(labels):,} rows)")
    print(f"[info] images dir: {config.IMAGES_DIR}")
    if "path_exists" in labels.columns:
        missing = int((~labels["path_exists"]).sum())
        if missing:
            print(f"[warn] {missing} labeled rows missing on disk")

    df, manifest = splits.build_splits(labels, cfg=cfg)
    out_dir = splits.save_splits(df, manifest)
    pipe = validation.ValidationPipeline(splits_dir=out_dir, rebuild=False)
    sanity = pipe.sanity_check()
    io.save_json("split_sanity.json", sanity)

    print(f"[done] splits -> {out_dir}")
    for split_name in config.SPLIT_NAMES:
        stats = manifest.split_stats[split_name]
        print(
            f"  {split_name:5s}: n={stats['n']:6,d}  "
            f"fraud={stats['n_fraud']:5,d}  "
            f"rate={stats['fraud_rate']:.3f}  "
            f"types={stats['n_types']:3d}  groups={stats['n_groups']:5,d}"
        )
    print(f"  cv folds: {len(manifest.cv_folds)}")
    if manifest.warnings:
        for w in manifest.warnings:
            print(f"  [warn] {w}")
    if sanity["duplicate_leakage_count"]:
        print(f"  [warn] duplicate leakage pairs: {sanity['duplicate_leakage_count']}")
    overlap = sanity["type_overlap"]
    for key, items in overlap.items():
        if items:
            print(f"  [warn] type overlap {key}: {len(items)} types")


if __name__ == "__main__":
    main()
