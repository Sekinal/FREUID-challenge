#!/usr/bin/env python
"""Self-training: pseudo-label the FREUID public_test (target domain) and retrain.

The public_test images are the actual leaderboard domain — far more on-domain
than any external dataset. We score them with an existing (ensemble) model, keep
only CONFIDENT predictions as pseudo-labels, mix them into the full FREUID train
set, and retrain. Then we produce a submission from the retrained model.

This is self-training (using the model's own confident predictions), NOT
reverse-engineering the hidden labels. Use conservative thresholds.

    .venv/bin/python scripts/24_pseudolabel.py \
        --teacher runs/f1_convnext_all/best.pt,runs/f2_dinov2_all/best.pt \
        --model convnext_base.fb_in22k_ft_in1k --img-size 384 \
        --aug domain --loss focal --epochs 4 --batch-size 128 --lr 5e-5 \
        --aux-dirs data/aux/idnet2025 --max-aux 40000 \
        --run-dir runs/pl_convnext --out submissions/sub_pseudolabel.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import aux_data, baseline, config, io  # noqa: E402

KEEP_COLS = [config.ID_COL, config.LABEL_COL, "abs_path", config.TYPE_COL]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pseudo-label public_test and retrain.")
    p.add_argument("--teacher", required=True, help="Comma-separated checkpoint(s) for pseudo-labels.")
    p.add_argument("--model", default="convnext_base.fb_in22k_ft_in1k")
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--aug", default="domain", choices=["none", "domain"])
    p.add_argument("--loss", default="focal", choices=["bce", "focal"])
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--conf-low", type=float, default=0.10, help="<= this -> pseudo genuine (0)")
    p.add_argument("--conf-high", type=float, default=0.90, help=">= this -> pseudo fraud (1)")
    p.add_argument("--aux-dirs", default=str(config.DATA_DIR / "aux" / "idnet2025"))
    p.add_argument("--max-aux", type=int, default=40000)
    p.add_argument("--run-dir", default=str(config.RUNS_DIR / "pl_convnext"))
    p.add_argument("--out", default=str(config.SUBMISSIONS_DIR / "sub_pseudolabel.csv"))
    return p.parse_args()


def public_test_frame() -> pd.DataFrame:
    root = config.PUBLIC_TEST_DIR
    rows = [{config.ID_COL: p.stem, "abs_path": str(p)} for p in sorted(root.rglob("*.jpeg"))]
    return pd.DataFrame(rows)


def predict_avg(checkpoints: list[str], frame: pd.DataFrame, batch_size: int) -> np.ndarray:
    device = baseline.get_device()
    order = list(frame[config.ID_COL].astype(str))
    acc = np.zeros(len(order), dtype=np.float64)
    for ckpt in checkpoints:
        model, cfg = baseline.load_model(Path(ckpt), device=device)
        cfg.batch_size = batch_size
        scores, ids = baseline.predict_frame(model, frame, cfg, device)
        by_id = dict(zip(ids, scores))
        acc += np.asarray([by_id[i] for i in order], dtype=np.float64)
        del model
        torch.cuda.empty_cache()
    return acc / len(checkpoints)


def main() -> None:
    args = parse_args()
    teachers = [c.strip() for c in args.teacher.split(",") if c.strip()]
    teachers = [t for t in teachers if Path(t).exists()]
    if not teachers:
        sys.exit("[error] no teacher checkpoints found")
    print(f"[info] teacher(s): {teachers}")

    # 1) score public_test, keep confident pseudo-labels
    pub = public_test_frame()
    print(f"[info] public_test: {len(pub):,} images")
    probs = predict_avg(teachers, pub, args.batch_size)
    pub = pub.assign(score=probs)
    pos = pub[pub["score"] >= args.conf_high].assign(**{config.LABEL_COL: 1})
    neg = pub[pub["score"] <= args.conf_low].assign(**{config.LABEL_COL: 0})
    pl = pd.concat([pos, neg], ignore_index=True)
    pl[config.TYPE_COL] = "PSEUDO/public_test"
    pl = pl[KEEP_COLS]
    print(f"[info] confident pseudo-labels: {len(pl):,} "
          f"(fraud={len(pos):,}, genuine={len(neg):,}) of {len(pub):,} "
          f"[{100*len(pl)/len(pub):.0f}% kept]")

    # 2) build train = full FREUID + aux + pseudo-labels
    full = io.filter_with_images(io.load_labels())[KEEP_COLS].copy()
    holdout = full.groupby(config.LABEL_COL, group_keys=False).sample(frac=0.05, random_state=args.seed)
    fr_train = full.drop(holdout.index)
    frames = [fr_train, pl]
    if args.max_aux > 0:
        roots = [d.strip() for d in args.aux_dirs.split(",") if d.strip()]
        aux = aux_data.load_idnet_frame(roots)[KEEP_COLS]
        if len(aux) > args.max_aux:
            aux = aux.sample(n=args.max_aux, random_state=args.seed).reset_index(drop=True)
        frames.append(aux)
        print(f"[info] aux: {len(aux):,}")
    train_df = pd.concat(frames, ignore_index=True)
    print(f"[info] combined train: {len(train_df):,} "
          f"(FREUID={len(fr_train):,} + pseudo={len(pl):,} + aux) "
          f"fraud_frac={float((train_df[config.LABEL_COL]==1).mean()):.3f}")

    # 3) retrain student
    cfg = baseline.BaselineConfig(
        model_name=args.model, img_size=args.img_size, batch_size=args.batch_size,
        epochs=args.epochs, lr=args.lr, num_workers=args.num_workers, seed=args.seed,
        aug=args.aug, loss_type=args.loss,
    )
    print(f"[info] training student: {args.model} aug={args.aug} loss={args.loss}")
    result = baseline.train_baseline(
        train_df, holdout, cfg=cfg, run_dir=Path(args.run_dir), test_df=None, prefiltered=True,
    )
    print(f"[info] student trained. holdout val FREUID={result['val']['freuid']:.4f}")

    # 4) submission from the retrained student
    model, scfg = baseline.load_model(Path(result["checkpoint_best"]))
    scfg.batch_size = args.batch_size
    scores, ids = baseline.predict_frame(model, pub, scfg, baseline.get_device())
    pred = pd.DataFrame({config.ID_COL: list(ids), config.LABEL_COL: scores})
    sub = io.load_sample_submission()
    out = sub[[config.ID_COL]].merge(pred, on=config.ID_COL, how="left")
    missing = int(out[config.LABEL_COL].isna().sum())
    if missing:
        fill = float(pred[config.LABEL_COL].median())
        out[config.LABEL_COL] = out[config.LABEL_COL].fillna(fill)
        print(f"[warn] filled {missing:,} missing ids with median {fill:.4f}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[done] wrote {out_path} ({len(out):,} rows). NOT uploaded.")


if __name__ == "__main__":
    main()
