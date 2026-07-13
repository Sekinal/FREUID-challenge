#!/usr/bin/env python
"""Train a fraud detector on FREUID train + auxiliary IDNet data.

Two modes:
  * sweep (default): train on the 3 FREUID train countries (+aux); honest
    hold-out scoring on val (Mauritius) and test (Mozambique).
  * --train-all: final submission model. Train on ~all labeled FREUID (+aux),
    holding out a small random in-domain slice only for checkpoint selection.

    .venv/bin/python scripts/13_train_with_aux.py \
        --model convnext_base.fb_in22k_ft_in1k --aug domain --loss focal \
        --aux-dirs data/aux/idnet2025,data/aux/idnet2025_scanned \
        --epochs 3 --batch-size 128 --img-size 384 --lr 5e-5 \
        --run-dir runs/e_convnext_aug_focal
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import aux_data, baseline, config, io, validation  # noqa: E402

KEEP_COLS = [config.ID_COL, config.LABEL_COL, "abs_path", config.TYPE_COL]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FREUID + auxiliary dataset.")
    p.add_argument("--model", default="convnext_base.fb_in22k_ft_in1k")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--aug", default="none", choices=["none", "domain"])
    p.add_argument("--loss", default="bce", choices=["bce", "focal"])
    p.add_argument("--aux-dirs", default=str(config.DATA_DIR / "aux" / "idnet2025"),
                   help="Comma-separated IDNet roots.")
    p.add_argument("--max-aux", type=int, default=30000,
                   help="Cap on auxiliary images (random sample). 0 = no aux.")
    p.add_argument("--train-all", action="store_true",
                   help="Final model: train on ~all FREUID countries + aux.")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--run-dir", default=str(config.RUNS_DIR / "convnext_aux"))
    p.add_argument("--rebuild-splits", action="store_true")
    return p.parse_args()


def load_aux(args) -> pd.DataFrame | None:
    if args.max_aux == 0:
        return None
    roots = [d.strip() for d in args.aux_dirs.split(",") if d.strip()]
    aux = aux_data.load_idnet_frame(roots)[KEEP_COLS]
    if args.max_aux and len(aux) > args.max_aux:
        aux = aux.sample(n=args.max_aux, random_state=args.seed).reset_index(drop=True)
    print(f"[info] IDNet aux: {len(aux):,} "
          f"(genuine={int((aux[config.LABEL_COL] == 0).sum()):,}, "
          f"fraud={int((aux[config.LABEL_COL] == 1).sum()):,}) from {roots}")
    return aux


def main() -> None:
    args = parse_args()
    pipe = validation.ValidationPipeline(rebuild=args.rebuild_splits)
    aux = load_aux(args)

    if args.train_all:
        full = io.filter_with_images(io.load_labels())[KEEP_COLS].copy()
        holdout = full.groupby(config.LABEL_COL, group_keys=False).sample(
            frac=0.06, random_state=args.seed)
        val_df = holdout
        test_df = None
        fr_train = full.drop(holdout.index)
        print(f"[info] train-all: full={len(full):,} train={len(fr_train):,} "
              f"holdout(val)={len(val_df):,}")
    else:
        fr_train = io.filter_with_images(pipe.train)[KEEP_COLS].copy()
        val_df = pipe.val
        test_df = pipe.test
        print(f"[info] FREUID train (3 countries): {len(fr_train):,}")

    frames = [fr_train] + ([aux] if aux is not None else [])
    train_df = pd.concat(frames, ignore_index=True)
    print(f"[info] combined train: {len(train_df):,} "
          f"(fraud frac={float((train_df[config.LABEL_COL] == 1).mean()):.3f})")

    cfg = baseline.BaselineConfig(
        model_name=args.model, img_size=args.img_size, batch_size=args.batch_size,
        epochs=args.epochs, lr=args.lr, num_workers=args.num_workers, seed=args.seed,
        amp=not args.no_amp, aug=args.aug, loss_type=args.loss,
    )
    print(f"[info] device: {baseline.get_device()}  model: {args.model}  "
          f"aug={args.aug}  loss={args.loss}")

    result = baseline.train_baseline(
        train_df, val_df, cfg=cfg, run_dir=Path(args.run_dir),
        test_df=test_df, prefiltered=True,
    )

    print("\n=== Results (lower FREUID = better) ===")
    v = result["val"]
    print(f"  val  FREUID={v['freuid']:.4f}  AuDET={v['audet']:.4f}  APCER@1%={v['apcer_at_1pct_bpcer']:.4f}")
    if "test" in result:
        t = result["test"]
        print(f"  test FREUID={t['freuid']:.4f}  AuDET={t['audet']:.4f}  APCER@1%={t['apcer_at_1pct_bpcer']:.4f}")
    print(f"[done] best checkpoint: {result['checkpoint_best']}")


if __name__ == "__main__":
    main()
