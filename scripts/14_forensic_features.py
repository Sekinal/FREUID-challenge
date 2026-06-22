#!/usr/bin/env python
"""Extract the hand-crafted forensic feature bank (freuid/forensics.py).

CPU-only, parallel over images. Writes a compact .npz the EDA script consumes.

    uv run scripts/14_forensic_features.py --sample 8000 --workers 24   # fast EDA pass
    uv run scripts/14_forensic_features.py                              # full dataset
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, forensics, io  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract forensic features.")
    p.add_argument("--sample", type=int, default=0,
                   help="stratified sample size by type x label (0 = all images)")
    p.add_argument("--workers", type=int, default=max(2, (os_cpu() - 2)))
    p.add_argument("--crop", type=int, default=forensics.CROP)
    p.add_argument("--out", default=str(config.ARTIFACTS_DIR / "forensic_features.npz"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk", type=int, default=64)
    return p.parse_args()


def os_cpu() -> int:
    import os
    return os.cpu_count() or 4


def stratified_sample(df, n: int, seed: int):
    """Sample ~n rows, balanced across (type, label) cells.

    Explicit per-group loop (pandas 3.0 groupby.apply drops the grouping columns).
    """
    n_cells = df[[config.TYPE_COL, config.LABEL_COL]].drop_duplicates().shape[0]
    per = max(1, n // max(1, n_cells))
    parts = [g.sample(min(len(g), per), random_state=seed)
             for _, g in df.groupby([config.TYPE_COL, config.LABEL_COL])]
    return pd.concat(parts).reset_index(drop=True)


def _worker(args):
    path, crop = args
    return forensics.feature_vector(path, crop=crop)


def main() -> None:
    args = parse_args()
    df = io.load_labels()
    df = df[df["path_exists"]].reset_index(drop=True)
    if args.sample and args.sample < len(df):
        df = stratified_sample(df, args.sample, args.seed)
    n = len(df)
    print(f"[features] extracting {forensics.N_FEATURES} features for {n:,} images "
          f"({args.workers} workers, crop={args.crop})")

    paths = df["abs_path"].astype(str).tolist()
    X = np.full((n, forensics.N_FEATURES), np.nan, dtype=np.float64)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, vec in enumerate(ex.map(_worker, ((p, args.crop) for p in paths),
                                       chunksize=args.chunk)):
            X[i] = vec
            done += 1
            if done % 2000 == 0 or done == n:
                rate = done / (time.time() - t0 + 1e-9)
                print(f"  {done:,}/{n:,}  ({rate:.0f} img/s, eta {(n-done)/rate:.0f}s)")

    ids = df[config.ID_COL].astype(str).to_numpy()
    labels = df[config.LABEL_COL].to_numpy().astype(np.int64)
    types = df[config.TYPE_COL].astype(str).to_numpy()
    is_digital = (df[config.IS_DIGITAL_COL].to_numpy().astype(bool)
                  if config.IS_DIGITAL_COL in df else np.ones(n, bool))

    nan_frac = float(np.isnan(X).mean())
    finite_rows = int(np.isfinite(X).all(axis=1).sum())
    print(f"[features] done in {time.time()-t0:.0f}s  "
          f"| NaN cells {nan_frac:.4f}  | fully-finite rows {finite_rows:,}/{n:,}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, ids=ids, labels=labels, types=types,
                        is_digital=is_digital,
                        feature_names=np.array(forensics.FEATURE_NAMES))
    print(f"[features] wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
