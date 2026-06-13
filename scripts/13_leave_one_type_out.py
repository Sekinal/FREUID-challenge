#!/usr/bin/env python
"""Leave-one-type-out (LOTO): the out-of-distribution generalization proxy.

For each document type, train on the other 4 types and score the held-out type.
This mirrors the private test's "unseen document type" axis (the competition's
real objective), which the in-distribution group-CV cannot measure. Splits are
group-aware (a near-dup component touching the held type is excluded from train).

Trains the capture-augmented baseline (no torch.compile -> avoids paying the
compile warmup 5x). Writes artifacts/loto_results.json.

    uv run scripts/13_leave_one_type_out.py
    uv run scripts/13_leave_one_type_out.py --epochs 3 --batch-size 512
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import baseline, config, io, splits, validation  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leave-one-type-out OOD evaluation.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--no-capture-aug", action="store_true")
    p.add_argument("--compile", action="store_true", help="Enable torch.compile (5x warmup).")
    p.add_argument("--max-train", type=int, default=None, help="Cap train samples per fold (smoke).")
    p.add_argument("--types", default=None, help="Comma-separated subset of held types (smoke).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipe = validation.ValidationPipeline(rebuild=False)
    table = pipe.table

    base_cfg = dict(
        img_size=args.img_size, batch_size=args.batch_size, epochs=args.epochs,
        num_workers=args.num_workers, amp=True, amp_dtype="bf16", channels_last=True,
        compile=args.compile, capture_aug=not args.no_capture_aug,
        max_train_samples=args.max_train,
    )
    print(f"[loto] config: {base_cfg}")

    only = {t.strip() for t in args.types.split(",")} if args.types else None
    results: dict[str, dict] = {}
    for held_type, train_df, val_df in splits.iter_type_holdout(table):
        if only is not None and held_type not in only:
            continue
        t0 = time.time()
        run_dir = config.RUNS_DIR / f"loto_{held_type.replace('/', '_')}"
        cfg = baseline.BaselineConfig(**base_cfg)
        print(f"\n[loto] hold={held_type}  train={len(train_df):,}  val={len(val_df):,}")
        res = baseline.train_baseline(
            train_df, val_df, cfg=cfg, run_dir=run_dir, test_df=None, save_artifact=False,
        )
        v = res["val"]
        v["n_val"] = len(val_df)
        v["n_train"] = len(train_df)
        v["minutes"] = round((time.time() - t0) / 60.0, 1)
        results[held_type] = v
        print(f"[loto] {held_type}: FREUID={v['freuid']:.4f} AuDET={v['audet']:.4f} "
              f"APCER@1%={v['apcer_at_1pct_bpcer']:.4f}  ({v['minutes']} min)")

    freuids = [r["freuid"] for r in results.values()]
    summary = {
        "per_type": results,
        "freuid_mean": float(np.mean(freuids)),
        "freuid_std": float(np.std(freuids)),
        "freuid_worst": float(np.max(freuids)),
        "n_types": len(freuids),
        "config": base_cfg,
    }
    out = io.save_json("loto_results.json", summary)

    print("\n=== Leave-one-type-out (OOD) — lower FREUID = better ===")
    for t, r in results.items():
        print(f"  {t:16s} FREUID={r['freuid']:.4f}  AuDET={r['audet']:.4f}  n_val={r['n_val']:,}")
    print(f"  {'MEAN':16s} FREUID={summary['freuid_mean']:.4f} ± {summary['freuid_std']:.4f}  "
          f"(worst {summary['freuid_worst']:.4f})")
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
