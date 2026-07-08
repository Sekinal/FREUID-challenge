#!/usr/bin/env python
"""Score public_test (target domain) with a checkpoint and emit confident pseudo-labels.

Writes a CSV (id,label,abs_path) of only the high-confidence public_test images,
to be mixed into training via scripts/30 --extra-csv. Self-training on the actual
competition domain (not off-domain noise). No-fusion only (feats ignored -> zeros).

    python3 scripts/27_make_pseudolabels.py --checkpoint runs/fusion_nofusion_30k.pt \
        --lo 0.05 --hi 0.95 --out artifacts/pseudo.csv
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--lo", type=float, default=0.05, help="<= lo -> pseudo genuine (0)")
    p.add_argument("--hi", type=float, default=0.95, help=">= hi -> pseudo fraud (1)")
    p.add_argument("--out", default="artifacts/pseudo.csv")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=12)
    return p.parse_args()


def public_test_frame():
    root = config.PUBLIC_TEST_DIR
    rows = [{config.ID_COL: p.stem, "abs_path": str(p), config.LABEL_COL: 0}
            for p in sorted(root.rglob("*")) if p.is_file()
            and p.suffix.lower() in config.IMAGE_EXTENSIONS]
    return pd.DataFrame(rows)


def predict_tta(model, frame, img_size, bs, workers, dev):
    feats = np.zeros((len(frame), fusion.FUSION_DIM), np.float32)
    tf0 = data.build_transforms(data.DataConfig(img_size=img_size, train=False))
    tf1 = T.Compose([T.Resize((img_size, img_size)), T.RandomHorizontalFlip(p=1.0),
                     T.ToTensor(), T.Normalize(data.IMAGENET_MEAN, data.IMAGENET_STD)])
    s = None
    for tf in (tf0, tf1):
        ld = DataLoader(fusion.FusionDataset(frame, feats, tf), batch_size=bs,
                        num_workers=workers, pin_memory=True)
        p = fusion._predict(model, ld, dev)
        s = p if s is None else s + p
    return s / 2.0


def main():
    args = parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    cfgd = ck["cfg"]
    model = fusion.FusionModel(cfgd.get("backbone", "tf_efficientnetv2_m.in21k_ft_in1k"),
                               use_fusion=False, pretrained=False).to(dev)
    model.load_state_dict(ck["model"]); model.eval()

    frame = public_test_frame()
    print(f"[pl] scoring public_test: {len(frame):,}")
    scores = predict_tta(model, frame, cfgd.get("img_size", 384), args.batch_size, args.workers, dev)

    frame = frame.assign(score=scores)
    pos = frame[frame.score >= args.hi].copy(); pos[config.LABEL_COL] = 1
    neg = frame[frame.score <= args.lo].copy(); neg[config.LABEL_COL] = 0
    pl = pd.concat([pos, neg], ignore_index=True)[[config.ID_COL, config.LABEL_COL, "abs_path"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pl.to_csv(args.out, index=False)
    print(f"[pl] kept {len(pl):,}/{len(frame):,} ({100*len(pl)/len(frame):.0f}%): "
          f"fraud={len(pos):,} genuine={len(neg):,} -> {args.out}")


if __name__ == "__main__":
    main()
