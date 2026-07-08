#!/usr/bin/env python
"""Print a compact table of all experiment results (runs/*/results.json).

    .venv/bin/python scripts/23_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config  # noqa: E402


def main() -> None:
    rows = []
    for rj in sorted(config.RUNS_DIR.glob("*/results.json")):
        try:
            d = json.loads(rj.read_text())
        except Exception:
            continue
        cfg = d.get("config", {})
        val = d.get("val", {}) or {}
        test = d.get("test", {}) or {}
        rows.append({
            "run": rj.parent.name,
            "model": cfg.get("model_name", "?"),
            "aug": cfg.get("aug", "?"),
            "loss": cfg.get("loss_type", "?"),
            "n_train": d.get("n_train", 0),
            "val_freuid": val.get("freuid"),
            "test_freuid": test.get("freuid"),
            "test_audet": test.get("audet"),
            "test_apcer": test.get("apcer_at_1pct_bpcer"),
        })

    if not rows:
        print("No results found under", config.RUNS_DIR)
        return

    def fmt(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else "  -   "

    print(f"{'run':<28}{'model':<34}{'aug':<8}{'loss':<6}{'ntrain':>8}"
          f"{'val':>9}{'TEST':>9}{'audet':>9}{'apcer':>9}")
    print("-" * 128)
    # sort by test_freuid (lower better); None last
    rows.sort(key=lambda r: (r["test_freuid"] is None, r["test_freuid"] if r["test_freuid"] is not None else 1e9))
    for r in rows:
        print(f"{r['run']:<28}{r['model']:<34}{r['aug']:<8}{r['loss']:<6}{r['n_train']:>8}"
              f"{fmt(r['val_freuid']):>9}{fmt(r['test_freuid']):>9}"
              f"{fmt(r['test_audet']):>9}{fmt(r['test_apcer']):>9}")
    print("\n(lower TEST FREUID = better cross-country generalization)")


if __name__ == "__main__":
    main()
