#!/usr/bin/env python
"""Teacher-ENSEMBLE pseudo-labels: cleaner labels than a single teacher.

Averages (probability) the heavy-TTA predictions of several strong checkpoints
on public_test, then keeps only confident ones as pseudo-labels. A better
teacher -> a better student than the 0.172 single-teacher PL.

    python3 scripts/32_ens_pseudolabels.py --teachers runs/a.pt,runs/b.pt \
        --lo 0.04 --hi 0.96 --out artifacts/pseudo_ens.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, data, fusion  # noqa: E402

FIXED = ("swin", "vit", "dinov2", "beit", "deit")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teachers", required=True)
    p.add_argument("--lo", type=float, default=0.04)
    p.add_argument("--hi", type=float, default=0.96)
    p.add_argument("--out", default="artifacts/pseudo_ens.csv")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=12)
    return p.parse_args()


def public_test_frame():
    root = config.PUBLIC_TEST_DIR
    rows = [{config.ID_COL: p.stem, "abs_path": str(p), config.LABEL_COL: 0}
            for p in sorted(root.rglob("*")) if p.is_file()
            and p.suffix.lower() in config.IMAGE_EXTENSIONS]
    return pd.DataFrame(rows)


def tta_prob(ckpt, frame, bs, workers, dev):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    cfgd = ck["cfg"]; bb = cfgd.get("backbone", "tf_efficientnetv2_m.in21k_ft_in1k")
    img = cfgd.get("img_size", 384)
    model = fusion.FusionModel(bb, use_fusion=False, pretrained=False).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    feats = np.zeros((len(frame), fusion.FUSION_DIM), np.float32)
    scales = [1.0] if any(k in bb for k in FIXED) else [1.0, 1.15]
    s, n = None, 0
    for sc in scales:
        size = img if sc == 1.0 else int(round(img * sc / 32) * 32)
        for flip in (False, True):
            tf = [T.Resize((size, size))] + ([T.RandomHorizontalFlip(1.0)] if flip else []) + \
                 [T.ToTensor(), T.Normalize(data.IMAGENET_MEAN, data.IMAGENET_STD)]
            ld = DataLoader(fusion.FusionDataset(frame, feats, T.Compose(tf)), batch_size=bs,
                            num_workers=workers, pin_memory=True)
            p = fusion._predict(model, ld, dev); s = p if s is None else s + p; n += 1
    return s / n


def main():
    args = parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    teachers = [c.strip() for c in args.teachers.split(",") if c.strip() and Path(c.strip()).exists()]
    print(f"[ens-pl] teachers: {teachers}")
    frame = public_test_frame()
    acc = np.zeros(len(frame))
    for t in teachers:
        acc += tta_prob(t, frame, args.batch_size, args.workers, dev)
        print(f"  [ens-pl] {t} done")
    scores = acc / len(teachers)
    frame = frame.assign(score=scores)
    pos = frame[frame.score >= args.hi].copy(); pos[config.LABEL_COL] = 1
    neg = frame[frame.score <= args.lo].copy(); neg[config.LABEL_COL] = 0
    pl = pd.concat([pos, neg], ignore_index=True)[[config.ID_COL, config.LABEL_COL, "abs_path"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pl.to_csv(args.out, index=False)
    print(f"[ens-pl] kept {len(pl):,}/{len(frame):,} ({100*len(pl)/len(frame):.0f}%): "
          f"fraud={len(pos):,} genuine={len(neg):,} -> {args.out}")


if __name__ == "__main__":
    main()
