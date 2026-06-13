#!/usr/bin/env python
"""Build leakage-safe train / val / test splits and CV fold assignments.

Strategy (default ``stratified_group``):
  - Group every image by its connected component in the exact+near duplicate
    graph (``artifacts/duplicates.json`` from scripts/05) so near-dups never
    cross a partition.
  - Assign whole groups to train/val/test and CV folds with StratifiedGroupKFold,
    balancing the ``type x label`` distribution.
  - Also emit leave-one-type-out (LOTO) metadata for a cross-type stress test.

Legacy ``--strategy type_holdout`` reproduces the old whole-type holdout.

Outputs under ``artifacts/splits/``: labeled_with_split.csv, {train,val,test}.csv,
manifest.json. Folds live in the ``cv_fold`` column (no giant id-list JSONs).

    uv run scripts/09_build_splits.py
    uv run scripts/09_build_splits.py --strategy type_holdout
    uv run scripts/09_build_splits.py --val-frac 0.15 --test-frac 0.15 --folds 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, io, splits, validation  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FREUID leakage-safe splits.")
    p.add_argument("--strategy", choices=("stratified_group", "type_holdout"), default="stratified_group")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--threshold", type=int, default=10, help="pHash near-dup threshold (metadata).")
    p.add_argument("--no-type-stratify", action="store_true", help="Stratify on label only.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = splits.SplitConfig(
        strategy=args.strategy,
        val_type_fraction=args.val_frac,
        test_type_fraction=args.test_frac,
        n_cv_folds=args.folds,
        near_dup_threshold=args.threshold,
        stratify_on_type=not args.no_type_stratify,
        random_state=args.seed,
    )

    labels = io.load_labels()
    if labels.empty:
        sys.exit("No labels found. Run scripts/00_download.py first.")

    print(f"[info] labels: {config.LABELS_CSV.name} ({len(labels):,} rows)  strategy={cfg.strategy}")
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
    print(f"  groups: {manifest.n_groups:,} ({manifest.n_nontrivial_groups:,} non-trivial / merged)")
    for split_name in config.SPLIT_NAMES:
        s = manifest.split_stats[split_name]
        rate = s["fraud_rate"]
        print(
            f"  {split_name:5s}: n={s['n']:7,d}  fraud={s['n_fraud']:6,d}  "
            f"rate={rate:.3f}  types={s['n_types']:2d}  groups={s['n_groups']:7,d}"
        )
    print(f"  cv folds: {len(manifest.cv_folds)}")
    if "cv_fraud_rate_spread" in sanity:
        sp = sanity["cv_fraud_rate_spread"]
        print(f"  cv fraud-rate balance: [{sp['min']:.3f}, {sp['max']:.3f}] std={sp['std']:.4f}")
    if manifest.type_holdout_folds:
        loto = ", ".join(f"{f['type']}({f['val_fraud_rate']:.2f})" for f in manifest.type_holdout_folds)
        print(f"  LOTO types: {loto}")

    for w in manifest.warnings:
        print(f"  [warn] {w}")
    if sanity["duplicate_leakage_count"]:
        print(f"  [warn] duplicate leakage pairs: {sanity['duplicate_leakage_count']}")
    else:
        print("  [ok] no duplicate-pair leakage across splits")
    for key, items in sanity["type_overlap"].items():
        if items:
            print(f"  [info] type overlap {key}: {len(items)} types (expected for stratified_group)")


if __name__ == "__main__":
    main()
