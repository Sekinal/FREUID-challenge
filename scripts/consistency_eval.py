"""Train a model on the OCR consistency features and evaluate it the way that
matters for FREUID: LEAVE-ONE-TYPE-OUT (train on 4 document types, test on the
unseen 5th) — the proxy for the private set's unseen document types. Also report
in-distribution stratified CV as a sanity check, plus single-feature baselines
and feature importances.
"""
import argparse, csv, sys
import numpy as np

sys.path.insert(0, "/root/freuid")
from freuid import consistency as C
from freuid.metrics import freuid_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def load(path):
    rows = list(csv.DictReader(open(path)))
    X = np.array([[float(r[k]) for k in C.FEATURE_NAMES] for r in rows], dtype=np.float64)
    y = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    t = np.array([r["type"] for r in rows])
    return X, y, t, rows


def mk():
    return HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.06, max_iter=400, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="/root/freuid/artifacts/consistency/features.csv")
    args = ap.parse_args()
    X, y, t, rows = load(args.feats)
    print(f"[data] {len(y)} rows, fraud rate {y.mean():.3f}, types {sorted(set(t))}\n")

    # ---- single-feature raw signal (no model) ----
    print("=== single-feature signal (in-distribution AUC) ===")
    aucs = []
    for j, name in enumerate(C.FEATURE_NAMES):
        col = X[:, j]
        if np.ptp(col) == 0:
            continue
        a = roc_auc_score(y, col)
        aucs.append((abs(a - 0.5), a, name))
    for _, a, name in sorted(aucs, reverse=True)[:10]:
        print(f"  {name:32s} AUC={a:.3f}")

    # ---- in-distribution stratified CV ----
    print("\n=== in-distribution 5-fold CV ===")
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        m = mk().fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    r = freuid_score(y, oof)
    print(f"  AUC={roc_auc_score(y, oof):.4f}  FREUID={r.freuid:.4f}  "
          f"AuDET={r.audet:.4f}  APCER@1%BPCER={r.apcer_at_1pct_bpcer:.4f}")

    # ---- LEAVE-ONE-TYPE-OUT (the OOD headline) ----
    print("\n=== LEAVE-ONE-TYPE-OUT (train on 4 types, test on unseen 5th) ===")
    types = sorted(set(t))
    frs = []
    for held in types:
        tr = t != held; te = t == held
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            print(f"  {held:16s} skipped (single-class)"); continue
        m = mk().fit(X[tr], y[tr]); p = m.predict_proba(X[te])[:, 1]
        r = freuid_score(y[te], p); auc = roc_auc_score(y[te], p)
        frs.append(r.freuid)
        print(f"  hold {held:14s} n={te.sum():4d}  AUC={auc:.3f}  "
              f"FREUID={r.freuid:.4f}  APCER@1%={r.apcer_at_1pct_bpcer:.3f}")
    if frs:
        print(f"\n  >>> mean LOTO FREUID = {np.mean(frs):.4f} "
              f"(baseline: EffNetV2+IDNet aux = 0.3556; ConvNeXt no-aux = 0.98)")

    # ---- feature importance (permutation on a holdout) ----
    print("\n=== permutation importance (in-dist holdout) ===")
    from sklearn.inspection import permutation_importance
    n = len(y); idx = np.random.RandomState(0).permutation(n); cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]
    m = mk().fit(X[tr], y[tr])
    pi = permutation_importance(m, X[te], y[te], n_repeats=8, random_state=0, scoring="roc_auc")
    for j in np.argsort(pi.importances_mean)[::-1][:10]:
        if pi.importances_mean[j] <= 0:
            continue
        print(f"  {C.FEATURE_NAMES[j]:32s} {pi.importances_mean[j]:.4f}")
    print("\nEVAL_DONE_MARKER")


if __name__ == "__main__":
    main()
