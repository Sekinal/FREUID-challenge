#!/usr/bin/env python
"""Policy-gradient (REINFORCE) fine-tuning for FREUID fraud detection.

Honest note: this is a high-variance baseline; supervised learning is the natural
fit for a labelled classification task. We warm-start from a good supervised
checkpoint and RL-finetune so it stays stable. Policy = sigmoid(logit) = P(fraud);
action ~ Bernoulli(p); reward = +1 if action==label else -1; REINFORCE with a
batch-mean baseline + small entropy bonus. Eval uses the continuous p (FREUID).

    python3 scripts/41_train_rl.py --init-from runs/fusion_nofusion_all_maeFT.pt \
        --backbone convnextv2_large.fcmae_ft_in22k_in1k --epochs 2 --save-name rl_maeFT
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
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import aux_data, config, data, fusion, io, metrics, validation  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="convnextv2_large.fcmae_ft_in22k_in1k")
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--init-from", required=True, help="warm-start (a good supervised model)")
    p.add_argument("--aux-roots", default="data/aux/idnet2025")
    p.add_argument("--max-aux", type=int, default=30000)
    p.add_argument("--extra-csv", default="artifacts/pseudo_ens.csv")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--save-name", required=True)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def resolve(f):
    o = io.attach_image_paths(f)
    return o[o["path_exists"]].reset_index(drop=True)


def build(args):
    pipe = validation.ValidationPipeline(rebuild=False)
    full = pd.concat([resolve(pipe.train), resolve(pipe.val), resolve(pipe.test)], ignore_index=True)
    val = full.sample(min(5000, len(full) // 10), random_state=0)
    train = full.drop(val.index).reset_index(drop=True); val = val.reset_index(drop=True)
    cols = [config.ID_COL, config.LABEL_COL, "abs_path"]
    parts = [train[cols]]
    if args.max_aux:
        aux = aux_data.load_idnet_frame([config.REPO_ROOT / r for r in args.aux_roots.split(",")])
        aux = aux[aux["abs_path"].map(lambda p: Path(p).exists())]
        if len(aux) > args.max_aux:
            aux = aux.sample(args.max_aux, random_state=0)
        parts.append(aux[cols])
    if args.extra_csv and Path(args.extra_csv).exists():
        ex = pd.read_csv(args.extra_csv); parts.append(ex[cols])
    train = pd.concat(parts, ignore_index=True)
    if args.smoke:
        train = train.sample(1200, random_state=0).reset_index(drop=True)
        val = val.sample(500, random_state=0).reset_index(drop=True); args.epochs = 1
    print(f"[rl] train={len(train):,} val={len(val):,}")
    return train, val


def main():
    args = parse_args()
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    train, val = build(args)
    zt = np.zeros((len(train), fusion.FUSION_DIM), np.float32)
    zv = np.zeros((len(val), fusion.FUSION_DIM), np.float32)
    tf_t = data.build_transforms(data.DataConfig(img_size=args.img_size, train=True))
    tf_e = data.build_transforms(data.DataConfig(img_size=args.img_size, train=False))
    tl = DataLoader(fusion.FusionDataset(train, zt, tf_t), batch_size=args.batch_size, shuffle=True,
                    drop_last=True, num_workers=args.workers, pin_memory=True, persistent_workers=True)
    vl = DataLoader(fusion.FusionDataset(val, zv, tf_e), batch_size=args.batch_size,
                    num_workers=args.workers, pin_memory=True)

    model = fusion.FusionModel(args.backbone, use_fusion=False).to(dev)
    ck = torch.load(args.init_from, map_location=dev, weights_only=False)
    miss, unexp = model.load_state_dict(ck["model"], strict=False)
    print(f"[rl] warm-start {args.init_from} (missing={len(miss)} unexpected={len(unexp)})")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = {"audet": math.inf}
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); rr = 0.0
        for bi, (x, ff, y) in enumerate(tl):
            x = x.to(dev, non_blocking=True); ff = ff.to(dev, non_blocking=True); y = y.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x, ff)
            p = torch.sigmoid(logit.float()).clamp(1e-4, 1 - 1e-4)
            dist = Bernoulli(probs=p)
            a = dist.sample()
            logp = dist.log_prob(a)
            reward = torch.where(a == y, 1.0, -1.0)
            adv = reward - reward.mean()
            loss = -(adv.detach() * logp).mean() - args.entropy * dist.entropy().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            rr += reward.mean().item()
            if bi % 100 == 0:
                print(f"  ep{ep} it{bi}/{len(tl)} avg_reward={rr/(bi+1):.3f} "
                      f"({(bi+1)*args.batch_size/(time.time()-t0):.0f} img/s)", flush=True)
        scores = fusion._predict(model, vl, dev)
        m = metrics.freuid_score(val[config.LABEL_COL].to_numpy(), scores)
        print(f"  [ep{ep}] val FREUID={m.freuid:.4f} AuDET={m.audet:.4f}")
        if m.audet < best["audet"]:
            best = {"epoch": ep, "audet": m.audet, "freuid": m.freuid}
            cfg = fusion.FusionConfig(backbone=args.backbone, img_size=args.img_size, use_fusion=False)
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__},
                       config.RUNS_DIR / f"{args.save_name}.pt")
            print(f"  [save] runs/{args.save_name}.pt (audet={m.audet:.4f})")
    print(f"[done] {args.save_name} best={best}")


if __name__ == "__main__":
    main()
