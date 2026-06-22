#!/usr/bin/env python
"""Ensemble one or more checkpoints, score locally, and build a Kaggle CSV.

Averages fraud probabilities across checkpoints (rank-changing -> can improve
FREUID, unlike calibration which is monotonic and leaves FREUID unchanged).
Reports the ensemble FREUID on the local FREUID test split (Mozambique) and
writes a submission CSV. Does NOT upload.

    .venv/bin/python scripts/22_ensemble_submission.py \
        --checkpoints runs/a/best.pt,runs/b/best.pt \
        --out submissions/ensemble.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import baseline, config, io, metrics, validation  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensemble + submission builder.")
    p.add_argument("--checkpoints", required=True, help="Comma-separated .pt paths.")
    p.add_argument("--out", default=str(config.SUBMISSIONS_DIR / "ensemble_submission.csv"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--no-score", action="store_true", help="Skip local FREUID scoring.")
    return p.parse_args()


def public_test_frame() -> pd.DataFrame | None:
    root = config.PUBLIC_TEST_DIR
    if not root.exists():
        return None
    rows = [{config.ID_COL: p.stem, "abs_path": str(p)} for p in sorted(root.rglob("*.jpeg"))]
    return pd.DataFrame(rows) if rows else None


def predict_avg(checkpoints: list[str], frame: pd.DataFrame, batch_size: int) -> np.ndarray:
    """Average sigmoid probabilities across checkpoints, aligned to frame order."""
    device = baseline.get_device()
    order = list(frame[config.ID_COL].astype(str))
    acc = np.zeros(len(order), dtype=np.float64)
    for ckpt in checkpoints:
        model, cfg = baseline.load_model(Path(ckpt), device=device)
        cfg.batch_size = batch_size
        scores, ids = baseline.predict_frame(model, frame, cfg, device)
        by_id = dict(zip(ids, scores))
        acc += np.asarray([by_id[i] for i in order], dtype=np.float64)
        print(f"  [predict] {ckpt}: {len(scores):,} scores")
        del model
        torch.cuda.empty_cache()
    return acc / len(checkpoints)


def main() -> None:
    args = parse_args()
    checkpoints = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    print(f"[info] ensembling {len(checkpoints)} checkpoint(s): {checkpoints}")

    if not args.no_score:
        pipe = validation.ValidationPipeline(rebuild=False)
        test = io.filter_with_images(pipe.test).reset_index(drop=True)
        scores = predict_avg(checkpoints, test, args.batch_size)
        res = metrics.freuid_score(test[config.LABEL_COL].to_numpy(), scores)
        print(f"[score] ensemble LOCAL TEST (Mozambique): FREUID={res.freuid:.4f} "
              f"AuDET={res.audet:.4f} APCER@1%={res.apcer_at_1pct_bpcer:.4f}")

    frame = public_test_frame()
    if frame is None:
        print("[warn] public_test not found; writing 0.5 submission")
        out = io.load_sample_submission()
        out[config.LABEL_COL] = 0.5
    else:
        print(f"[info] scoring public_test ({len(frame):,} images)")
        probs = predict_avg(checkpoints, frame, args.batch_size)
        pred = pd.DataFrame({config.ID_COL: frame[config.ID_COL].astype(str), config.LABEL_COL: probs})
        sub = io.load_sample_submission()
        out = sub[[config.ID_COL]].merge(pred, on=config.ID_COL, how="left")
        missing = int(out[config.LABEL_COL].isna().sum())
        if missing:
            fill = float(pred[config.LABEL_COL].median())
            print(f"[warn] {missing:,} ids without local images; filling with median {fill:.4f}")
            out[config.LABEL_COL] = out[config.LABEL_COL].fillna(fill)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[done] wrote {out_path} ({len(out):,} rows). NOT uploaded.")


if __name__ == "__main__":
    main()
