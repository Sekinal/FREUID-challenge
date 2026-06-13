#!/usr/bin/env python
"""Smoke-test metrics + validation pipeline on saved splits.

Uses a trivial score baseline (constant 0.5) to verify wiring only.

    uv run scripts/10_validate_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, validation  # noqa: E402


def main() -> None:
    pipe = validation.ValidationPipeline(rebuild=True)
    sanity = pipe.sanity_check()
    print("[sanity]", sanity["partitions"])

    def dummy_scores(train, val):
        _ = train
        return np.full(len(val), 0.5)

    report = pipe.run_cv(dummy_scores)
    print("[cv summary]", report.cv_summary)
    path = pipe.save_report(report, "validation_smoke.json")
    print(f"[done] wrote {path}")


if __name__ == "__main__":
    main()
