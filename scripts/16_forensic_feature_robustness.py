#!/usr/bin/env python
"""Per-feature *within-type* discriminative robustness for the forensic bank.

The LOTO GBM (scripts/15) collapses on unseen types (mean AUC < 0.5): the model
learns type-specific *absolute* thresholds that invert on a new issuer/pipeline.
This asks a finer question: which individual features separate fraud/genuine
*within every type* (i.e. carry signal that is real regardless of the type's
baseline level)? Those are the only credible candidates to fuse into the CNN,
where the backbone can supply the type context the raw feature lacks.

For each feature we compute a direction-agnostic ROC-AUC inside each type, then
report the mean and (crucially) the MIN across types. High MIN = universally
discriminative; near-0.5 MIN = only works by proxying the type.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, io  # noqa: E402

from sklearn.metrics import roc_auc_score  # noqa: E402


def main() -> None:
    npz = config.ARTIFACTS_DIR / "forensic_features.npz"
    data = np.load(npz, allow_pickle=True)
    X = np.where(np.isfinite(data["X"].astype(np.float64)), data["X"].astype(np.float64), np.nan)
    y = data["labels"].astype(np.int64)
    types = data["types"].astype(str)
    names = list(data["feature_names"].astype(str))
    uniq = sorted(set(types))

    rows = []
    for j, nm in enumerate(names):
        col = X[:, j]
        per_type = []
        for t in uniq:
            m = (types == t) & np.isfinite(col)
            yy, ss = y[m], col[m]
            if len(np.unique(yy)) < 2:
                continue
            a = roc_auc_score(yy, ss)
            per_type.append(max(a, 1.0 - a))  # direction-agnostic within type
        if not per_type:
            continue
        rows.append((nm, float(np.mean(per_type)), float(np.min(per_type)),
                     float(np.max(per_type))))

    rows.sort(key=lambda r: -r[2])  # by MIN within-type AUC (robustness)
    print(f"{'feature':<24} {'mean':>6} {'MIN':>6} {'max':>6}   (within-type, direction-agnostic)")
    for nm, mean, mn, mx in rows:
        flag = "  <-- robust" if mn >= 0.58 else ""
        print(f"{nm:<24} {mean:6.3f} {mn:6.3f} {mx:6.3f}{flag}")

    robust = [r[0] for r in rows if r[1] >= 0.58]
    print(f"\nfeatures with MIN within-type AUC >= 0.58: "
          f"{[r[0] for r in rows if r[1] >= 0.58 and r[2] >= 0.58]}")
    io.save_json("forensic_feature_robustness.json",
                 {"by_min_within_type_auc": [
                     {"name": nm, "mean": mean, "min": mn, "max": mx}
                     for nm, mean, mn, mx in rows]})
    print("[robustness] wrote artifacts/forensic_feature_robustness.json")


if __name__ == "__main__":
    main()
