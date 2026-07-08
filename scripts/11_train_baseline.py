#!/usr/bin/env python
"""Train the EfficientNet baseline and report FREUID scores.

Uses the leakage-safe group splits from scripts/09_build_splits.py. Tuned to
saturate the A100: bf16 autocast (no GradScaler), channels_last, torch.compile,
and a wide dataloader. Best checkpoint is chosen by AuDET (smooth), not FREUID.

    uv run scripts/11_train_baseline.py
    uv run scripts/11_train_baseline.py --epochs 3 --batch-size 512 --img-size 384
    uv run scripts/11_train_baseline.py --max-train 2000        # quick smoke
    uv run scripts/11_train_baseline.py --amp-dtype fp16 --no-compile
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import baseline, config, validation  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FREUID EfficientNet baseline.")
    p.add_argument("--model", default="efficientnet_b2")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train", type=int, default=None, help="Optional cap for quick runs.")
    p.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-channels-last", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--run-dir", default=str(config.RUNS_DIR / "baseline"))
    p.add_argument("--rebuild-splits", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipe = validation.ValidationPipeline(rebuild=args.rebuild_splits)
    cfg = baseline.BaselineConfig(
        model_name=args.model,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=not args.no_amp,
        amp_dtype=args.amp_dtype,
        channels_last=not args.no_channels_last,
        compile=not args.no_compile,
        max_train_samples=args.max_train,
    )

    device = baseline.get_device()
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"[gpu] {props.name}  {props.total_memory/1e9:.0f} GB  "
              f"amp={cfg.amp_dtype if cfg.amp else 'off'}  channels_last={cfg.channels_last}  "
              f"compile={cfg.compile}  batch={cfg.batch_size}  workers={cfg.num_workers}")
    print(f"[info] train={len(pipe.train):,} val={len(pipe.val):,} test={len(pipe.test):,}")

    result = baseline.train_baseline(
        pipe.train, pipe.val, cfg=cfg, run_dir=Path(args.run_dir), test_df=pipe.test,
    )

    print("\n=== Baseline results (lower FREUID = better) ===")
    print(f"  val  FREUID={result['val']['freuid']:.4f}  AuDET={result['val']['audet']:.4f}  "
          f"APCER@1%BPCER={result['val']['apcer_at_1pct_bpcer']:.4f}")
    if "test" in result:
        print(f"  test FREUID={result['test']['freuid']:.4f}  AuDET={result['test']['audet']:.4f}  "
              f"APCER@1%BPCER={result['test']['apcer_at_1pct_bpcer']:.4f}")
    print(f"[done] checkpoint: {result['checkpoint_best']}")
    print(f"[done] results: {config.ARTIFACTS_DIR / 'baseline_results.json'}")


if __name__ == "__main__":
    main()
