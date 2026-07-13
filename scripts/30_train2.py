#!/usr/bin/env python
"""Flexible no-fusion trainer reusing freuid.fusion internals.

Supports the data-plan levers without touching scripts/24:
  --extra-csv     mix an extra labeled frame (pseudo-labels / FantasyID)
  --init-from     warm-start weights from a checkpoint (2-stage pretrain->finetune)
  --lr            override LR (low LR for stage-B finetune)
  --aux-roots/--max-aux   IDNet aux mixing (0 max-aux = use ALL)

    python3 scripts/30_train2.py --train-all --aux --max-aux 30000 \
        --extra-csv artifacts/pseudo.csv --save-name fusion_nofusion_all_pl
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import aux_data, config, data, fusion, io, metrics, validation  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="tf_efficientnetv2_m.in21k_ft_in1k")
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--loss", default="focal", choices=["focal", "bce", "rank"])
    p.add_argument("--aug-strength", default="default", choices=["default", "heavy"])
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--train-all", action="store_true")
    p.add_argument("--loto-type", default="", help="hold this FREUID type out as cross-country test")
    p.add_argument("--aux", action="store_true")
    p.add_argument("--aux-roots", default="data/aux/idnet2025")
    p.add_argument("--max-aux", type=int, default=30000, help="0 = use all aux")
    p.add_argument("--extra-csv", default="", help="extra labeled frame: id,label,abs_path")
    p.add_argument("--max-extra", type=int, default=0, help="cap on extra-csv rows (0=all)")
    p.add_argument("--init-from", default="", help="warm-start weights from checkpoint")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="linear-probe: freeze backbone, train only the head")
    p.add_argument("--save-name", required=True)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def rank_loss(logit, y):
    """Differentiable ranking surrogate (soft-AUC): push fraud logits above
    genuine logits via pairwise logistic loss, aligning with rank-based FREUID.
    Small BCE term keeps it calibrated/stable."""
    import torch.nn.functional as F
    pos = logit[y > 0.5]; neg = logit[y < 0.5]
    bce = F.binary_cross_entropy_with_logits(logit, y)
    if pos.numel() == 0 or neg.numel() == 0:
        return bce
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)   # want pos (fraud) > neg (genuine)
    rank = F.softplus(-diff).mean()
    return rank + 0.2 * bce


def resolve(frame):
    out = io.attach_image_paths(frame)
    return out[out["path_exists"]].reset_index(drop=True)


def build_train(args):
    pipe = validation.ValidationPipeline(rebuild=False)
    train, val, test = resolve(pipe.train), resolve(pipe.val), resolve(pipe.test)
    full = pd.concat([train, val, test], ignore_index=True)
    test = None
    if args.loto_type:
        # hold one country out as a clean cross-country test (the LB proxy)
        test = full[full[config.TYPE_COL] == args.loto_type].reset_index(drop=True)
        pool = full[full[config.TYPE_COL] != args.loto_type].reset_index(drop=True)
        val = pool.sample(min(5000, max(1, len(pool) // 10)), random_state=0)
        train = pool.drop(val.index).reset_index(drop=True)
        val = val.reset_index(drop=True)
        print(f"[loto] held out {args.loto_type}: test={len(test):,} train={len(train):,} val={len(val):,}")
    elif args.train_all:
        val = full.sample(min(5000, max(1, len(full) // 10)), random_state=0)
        train = full.drop(val.index).reset_index(drop=True)
        val = val.reset_index(drop=True)
    cols = [config.ID_COL, config.LABEL_COL, "abs_path"]
    parts = [train[cols]]

    if args.aux:
        roots = [config.REPO_ROOT / r for r in args.aux_roots.split(",")]
        aux = aux_data.load_idnet_frame(roots)
        aux = aux[aux["abs_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
        if args.max_aux and len(aux) > args.max_aux:
            aux = aux.sample(args.max_aux, random_state=0).reset_index(drop=True)
        parts.append(aux[cols])
        print(f"[aux] +{len(aux):,} IDNet from {args.aux_roots}")

    if args.extra_csv:
        ex = pd.read_csv(args.extra_csv)
        ex = ex[ex["abs_path"].map(lambda p: Path(str(p)).exists())].reset_index(drop=True)
        if args.max_extra and len(ex) > args.max_extra:
            ex = ex.sample(args.max_extra, random_state=0).reset_index(drop=True)
        parts.append(ex[cols])
        print(f"[extra] +{len(ex):,} rows from {args.extra_csv} "
              f"(fraud={int(ex[config.LABEL_COL].sum()):,})")

    train = pd.concat(parts, ignore_index=True)
    if args.smoke:
        train = train.sample(min(len(train), 1500), random_state=0).reset_index(drop=True)
        val = val.sample(min(len(val), 600), random_state=0).reset_index(drop=True)
        if test is not None:
            test = test.sample(min(len(test), 600), random_state=0).reset_index(drop=True)
        args.epochs = 1
    print(f"[data] train={len(train):,} val={len(val):,} "
          f"test={0 if test is None else len(test):,} "
          f"fraud-rate={train[config.LABEL_COL].mean():.3f}")
    return train, val, (test[cols].reset_index(drop=True) if test is not None else None)


def main():
    args = parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    train, val, test = build_train(args)
    zfeat_tr = np.zeros((len(train), fusion.FUSION_DIM), np.float32)
    zfeat_va = np.zeros((len(val), fusion.FUSION_DIM), np.float32)

    cfg = fusion.FusionConfig(backbone=args.backbone, img_size=args.img_size,
                              epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                              use_fusion=False, loss_type=args.loss,
                              aug_strength=args.aug_strength, num_workers=args.workers)

    tf_train = data.build_transforms(data.DataConfig(img_size=cfg.img_size, train=True,
                                                     capture_aug=cfg.capture_aug,
                                                     aug_strength=cfg.aug_strength))
    tf_eval = data.build_transforms(data.DataConfig(img_size=cfg.img_size, train=False))
    tr_loader = DataLoader(fusion.FusionDataset(train, zfeat_tr, tf_train),
                           batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                           num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True)
    va_loader = DataLoader(fusion.FusionDataset(val, zfeat_va, tf_eval),
                           batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    te_loader = None
    if test is not None:
        zfeat_te = np.zeros((len(test), fusion.FUSION_DIM), np.float32)
        te_loader = DataLoader(fusion.FusionDataset(test, zfeat_te, tf_eval),
                               batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)

    model = fusion.FusionModel(cfg.backbone, use_fusion=False).to(dev)
    if args.init_from:
        ck = torch.load(args.init_from, map_location=dev, weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"[init] warm-started from {args.init_from} "
              f"(missing={len(missing)} unexpected={len(unexpected)})")
    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[freeze] backbone frozen (linear-probe); trainable params={n_train:,}")
    train_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * len(tr_loader))
    loss_fn = rank_loss if args.loss == "rank" else fusion._loss_fn(cfg)

    best = {"audet": math.inf}
    for ep in range(cfg.epochs):
        model.train(); t0 = time.time(); run = 0.0
        for bi, (x, ff, y) in enumerate(tr_loader):
            x = x.to(dev, non_blocking=True); ff = ff.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                logit = model(x, ff); loss = loss_fn(logit, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            run += loss.item()
            if bi % 100 == 0:
                print(f"  ep{ep} it{bi}/{len(tr_loader)} loss={run/(bi+1):.4f} "
                      f"({(bi+1)*cfg.batch_size/(time.time()-t0):.0f} img/s)", flush=True)
        scores = fusion._predict(model, va_loader, dev)
        m = metrics.freuid_score(val[config.LABEL_COL].to_numpy(), scores)
        tline = ""
        test_freuid = None
        if te_loader is not None:
            ts = fusion._predict(model, te_loader, dev)
            tm = metrics.freuid_score(test[config.LABEL_COL].to_numpy(), ts)
            test_freuid = tm.freuid
            tline = f" || LOTO-test FREUID={tm.freuid:.4f} AuDET={tm.audet:.4f}"
        print(f"  [ep{ep}] val FREUID={m.freuid:.4f} AuDET={m.audet:.4f}{tline}")
        if m.audet < best["audet"]:
            best = {"epoch": ep, "audet": m.audet, "freuid": m.freuid, "loto_test_freuid": test_freuid}
            config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__},
                       config.RUNS_DIR / f"{args.save_name}.pt")
            print(f"  [save] runs/{args.save_name}.pt (val audet={m.audet:.4f})")
    # PROXY line: the cross-country FREUID at the val-selected checkpoint
    if best.get("loto_test_freuid") is not None:
        print(f"[PROXY] {args.save_name} loto={args.loto_type} CROSS-COUNTRY-FREUID={best['loto_test_freuid']:.4f}")
    print(f"[done] {args.save_name} best={best}")


if __name__ == "__main__":
    main()
