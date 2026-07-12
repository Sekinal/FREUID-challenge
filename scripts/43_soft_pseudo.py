#!/usr/bin/env python
"""Soft pseudo-labels: keep ALL public_test with the teacher's continuous score
as a soft target (no thresholding, no discarding the uncertain middle).

Trained later with BCE (which accepts soft targets in [0,1]) -> extracts more
transductive signal from the target domain than hard 0/1 pseudo-labels.

    python3 scripts/43_soft_pseudo.py --teacher runs/fusion_nofusion_all_maeFT.pt \
        --out artifacts/soft_pseudo.csv
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, data, fusion  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="runs/fusion_nofusion_all_maeFT.pt")
    p.add_argument("--out", default="artifacts/soft_pseudo.csv")
    p.add_argument("--temp", type=float, default=1.0, help="sharpen exponent (>1 sharpens)")
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--workers", type=int, default=12)
    return p.parse_args()


def public_test_frame():
    root = config.PUBLIC_TEST_DIR
    rows = [{config.ID_COL: p.stem, "abs_path": str(p), config.LABEL_COL: 0}
            for p in sorted(root.rglob("*")) if p.is_file()
            and p.suffix.lower() in config.IMAGE_EXTENSIONS]
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    dev = "cuda"
    ck = torch.load(args.teacher, map_location=dev, weights_only=False)
    cfgd = ck["cfg"]; img = cfgd.get("img_size", 384)
    model = fusion.FusionModel(cfgd.get("backbone", "tf_efficientnetv2_m.in21k_ft_in1k"),
                               use_fusion=False, pretrained=False).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    frame = public_test_frame()
    feats = np.zeros((len(frame), fusion.FUSION_DIM), np.float32)
    s = None
    for flip in (False, True):
        tf = [T.Resize((img, img))] + ([T.RandomHorizontalFlip(1.0)] if flip else []) + \
             [T.ToTensor(), T.Normalize(data.IMAGENET_MEAN, data.IMAGENET_STD)]
        ld = DataLoader(fusion.FusionDataset(frame, feats, T.Compose(tf)), batch_size=args.batch_size,
                        num_workers=args.workers, pin_memory=True)
        p = fusion._predict(model, ld, dev); s = p if s is None else s + p
    soft = (s / 2.0).clip(1e-4, 1 - 1e-4)
    if args.temp != 1.0:   # optional sharpening toward 0/1
        soft = soft ** args.temp / (soft ** args.temp + (1 - soft) ** args.temp)
    out = frame[[config.ID_COL, "abs_path"]].copy()
    out[config.LABEL_COL] = soft
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out[[config.ID_COL, config.LABEL_COL, "abs_path"]].to_csv(args.out, index=False)
    print(f"[soft] {len(out):,} soft labels (mean={soft.mean():.3f}, "
          f">0.5={int((soft>0.5).sum()):,}) -> {args.out}")


if __name__ == "__main__":
    main()
