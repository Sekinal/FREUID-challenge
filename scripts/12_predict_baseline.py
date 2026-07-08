#!/usr/bin/env python
"""Generate a Kaggle submission CSV from a trained baseline checkpoint.

Scores ``public_test`` images when available; otherwise writes a dummy submit
from ``sample_submission`` ids with model scores only for labeled train ids
(for pipeline testing).

    uv run scripts/12_predict_baseline.py --checkpoint runs/baseline/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import baseline, config, io  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict baseline submission CSV.")
    p.add_argument("--checkpoint", default=str(config.RUNS_DIR / "baseline" / "best.pt"))
    p.add_argument("--out", default=str(config.SUBMISSIONS_DIR / "baseline_submission.csv"))
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--fill", type=float, default=0.5,
                   help="Constant for private ids without local images. Ignored by the "
                        "public LB (private dummies are not scored), so a fixed constant "
                        "keeps submissions comparable across runs.")
    return p.parse_args()


def _public_test_frame() -> pd.DataFrame | None:
    root = config.PUBLIC_TEST_DIR
    if not root.exists():
        return None
    rows = []
    for path in sorted(root.rglob("*.jpeg")):
        rows.append({config.ID_COL: path.stem, "abs_path": str(path)})
    if not rows:
        return None
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    device = baseline.get_device()
    model, cfg = baseline.load_model(Path(args.checkpoint), device=device)
    if args.batch_size:
        cfg.batch_size = args.batch_size

    test_frame = _public_test_frame()
    if test_frame is not None:
        print(f"[info] scoring public_test ({len(test_frame):,} images)")
        scores, ids = baseline.predict_frame(model, test_frame, cfg, device)
        pred = pd.DataFrame({config.ID_COL: ids, config.LABEL_COL: scores})
        sub = io.load_sample_submission()
        out = sub[[config.ID_COL]].merge(pred, on=config.ID_COL, how="left")
        missing = int(out[config.LABEL_COL].isna().sum())
        if missing:
            print(f"[warn] {missing} private ids without local images; filling with "
                  f"constant {args.fill} (not scored on the public LB)")
            out[config.LABEL_COL] = out[config.LABEL_COL].fillna(args.fill)
    else:
        print("[warn] public_test not found; writing sample_submission with 0.5 scores")
        out = io.load_sample_submission()
        out[config.LABEL_COL] = 0.5

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[done] {out_path} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
