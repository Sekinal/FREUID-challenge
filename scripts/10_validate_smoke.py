#!/usr/bin/env python
"""Smoke-test the validation pipeline on saved splits.

Exercises every new path with trivial scorers (no model): leakage-failing
sanity check, group CV, hold-out scoring with bootstrap CIs + per-type
breakdown, and leave-one-type-out. Verifies wiring, not model quality.

    uv run scripts/10_validate_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, validation  # noqa: E402


def main() -> None:
    pipe = validation.ValidationPipeline(rebuild=False, bootstrap=200)

    sanity = pipe.sanity_check()  # raises on any leakage
    print(f"[sanity] strategy={sanity['strategy']} groups={sanity['n_groups']:,} "
          f"(non-trivial={sanity['n_nontrivial_groups']:,}) "
          f"dup_leakage={sanity['duplicate_leakage_count']}")
    if "cv_fraud_rate_spread" in sanity:
        sp = sanity["cv_fraud_rate_spread"]
        print(f"[sanity] cv fraud-rate spread std={sp['std']:.4f} over {sp['folds']} folds")

    # A weak-but-nontrivial scorer (fraud slightly more likely for is_digital):
    rng = np.random.default_rng(0)

    def score_fn(frame):
        return rng.random(len(frame))

    def fit_predict(train, val):
        return rng.random(len(val))

    cv = pipe.run_cv(fit_predict)
    print(f"[cv] {cv.cv_summary}")

    holdout = pipe.evaluate_holdout(score_fn, splits_to_score=("val",))
    val_entry = holdout.holdout["val"]
    ci = val_entry.get("ci", {}).get("freuid", {})
    print(f"[holdout val] freuid={val_entry['freuid']:.4f} "
          f"CI=[{ci.get('lo', float('nan')):.4f}, {ci.get('hi', float('nan')):.4f}]")
    print(f"[per-type val] {list(holdout.per_type.get('val', {}).keys())}")

    loto = pipe.run_leave_one_type_out(fit_predict)
    summary = next((e for e in loto if e.get("type") == "__summary__"), {})
    print(f"[loto] freuid_mean={summary.get('freuid_mean'):.4f} over {summary.get('n_types')} types")

    holdout.loto = loto
    path = pipe.save_report(holdout, "validation_smoke.json")
    print(f"[done] wrote {path}")


if __name__ == "__main__":
    main()
