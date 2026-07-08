#!/usr/bin/env python
"""Discriminative-power EDA for the forensic feature bank + leave-one-type-out GBM.

Answers the question "is there standalone signal in hand-crafted forensics, and
does it *generalise to unseen document types*?" — the axis the competition rewards.

- Per-feature ROC-AUC + mutual information (overall and within each type).
- Leave-one-type-out (LOTO) gradient-boosted trees scored with the real FREUID
  metric: train on 4 types, predict the held-out 5th. This is the OOD headline.
- In-distribution 5-fold CV for reference (to expose the in-dist -> OOD gap).

    uv run scripts/15_forensic_eda.py
    uv run scripts/15_forensic_eda.py --features artifacts/forensic_features.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, io, metrics  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.feature_selection import mutual_info_classif  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forensic feature discriminative EDA.")
    p.add_argument("--features", default=str(config.ARTIFACTS_DIR / "forensic_features.npz"))
    p.add_argument("--out", default=str(config.ARTIFACTS_DIR / "forensic_eda.json"))
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def clean(X: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(X), X, np.nan)


def median_impute(X: np.ndarray) -> np.ndarray:
    Xi = X.copy()
    med = np.nanmedian(Xi, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    idx = np.where(np.isnan(Xi))
    Xi[idx] = np.take(med, idx[1])
    return Xi


def new_gbm(args) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=args.max_iter, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=args.seed,
    )


def safe_auc(y, s) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def main() -> None:
    args = parse_args()
    data = np.load(args.features, allow_pickle=True)
    X = clean(data["X"].astype(np.float64))
    y = data["labels"].astype(np.int64)
    types = data["types"].astype(str)
    names = list(data["feature_names"].astype(str))
    n, d = X.shape
    print(f"[eda] {n:,} images x {d} features; fraud-rate {y.mean():.3f}; "
          f"types {sorted(set(types))}")

    Xi = median_impute(X)

    # ---- per-feature discriminative power (overall) ----
    mi = mutual_info_classif(Xi, y, random_state=args.seed)
    per_feature = []
    for j, nm in enumerate(names):
        raw = safe_auc(y, Xi[:, j])
        auc = max(raw, 1.0 - raw) if raw == raw else float("nan")  # direction-agnostic
        per_feature.append({"name": nm, "auc": auc, "auc_raw": raw,
                            "direction": ("fraud_high" if raw >= 0.5 else "fraud_low"),
                            "mutual_info": float(mi[j])})
    per_feature.sort(key=lambda r: (-(r["auc"] if r["auc"] == r["auc"] else 0)))
    top = [r["name"] for r in per_feature[:15]]
    print("\n[eda] top features by |AUC-0.5| (overall, in-distribution):")
    for r in per_feature[:15]:
        print(f"  {r['name']:<24} AUC={r['auc']:.3f} ({r['direction']:<10}) MI={r['mutual_info']:.4f}")

    # ---- leave-one-type-out GBM (OOD headline) ----
    print("\n[eda] leave-one-type-out GBM (train 4 types -> predict held-out type):")
    loto = {}
    fr_list, auc_list = [], []
    for t in sorted(set(types)):
        tr = types != t
        te = ~tr
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = new_gbm(args).fit(X[tr], y[tr])  # HistGB handles NaN natively
        proba = clf.predict_proba(X[te])[:, 1]
        m = metrics.freuid_score(y[te], proba)
        auc = safe_auc(y[te], proba)
        loto[t] = {"freuid": m.freuid, "audet": m.audet,
                   "apcer_at_1pct_bpcer": m.apcer_at_1pct_bpcer, "auc": auc,
                   "n_test": int(te.sum()), "fraud_rate": float(y[te].mean())}
        fr_list.append(m.freuid)
        auc_list.append(auc)
        print(f"  hold-out {t:<16} FREUID={m.freuid:.4f}  AuDET={m.audet:.4f}  "
              f"APCER@1%={m.apcer_at_1pct_bpcer:.4f}  AUC={auc:.3f}  (n={te.sum():,})")
    loto_mean_freuid = float(np.mean(fr_list)) if fr_list else float("nan")
    loto_mean_auc = float(np.mean(auc_list)) if auc_list else float("nan")
    print(f"  --> LOTO mean FREUID={loto_mean_freuid:.4f}  mean AUC={loto_mean_auc:.3f}")

    # ---- in-distribution 5-fold CV (reference / ceiling) ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fr_cv, auc_cv = [], []
    for tr_idx, te_idx in skf.split(X, y):
        clf = new_gbm(args).fit(X[tr_idx], y[tr_idx])
        proba = clf.predict_proba(X[te_idx])[:, 1]
        fr_cv.append(metrics.freuid_score(y[te_idx], proba).freuid)
        auc_cv.append(safe_auc(y[te_idx], proba))
    cv_freuid = float(np.mean(fr_cv))
    cv_auc = float(np.mean(auc_cv))
    print(f"\n[eda] in-distribution 5-fold CV: FREUID={cv_freuid:.4f}  AUC={cv_auc:.3f}")
    print(f"[eda] OOD gap (LOTO - in-dist FREUID): {loto_mean_freuid - cv_freuid:+.4f}  "
          f"(large gap = features memorise type-specific cues, not universal forensics)")

    report = {
        "n_images": int(n), "n_features": int(d), "fraud_rate": float(y.mean()),
        "top_features": top,
        "per_feature": per_feature,
        "loto": {"per_type": loto, "mean_freuid": loto_mean_freuid,
                 "mean_auc": loto_mean_auc},
        "in_distribution_cv": {"freuid": cv_freuid, "auc": cv_auc},
        "ood_gap_freuid": loto_mean_freuid - cv_freuid,
    }
    out = io.save_json(Path(args.out).name, report)
    print(f"\n[eda] wrote {out}")


if __name__ == "__main__":
    main()
