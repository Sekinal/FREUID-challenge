#!/usr/bin/env python
"""Average TTA predictions across checkpoints on public_test -> submission CSV.

Ensembling distinct models changes the score ranking (unlike calibration) and
usually reduces variance -> can beat any single model. No-fusion (feats=zeros).

    python3 scripts/28_ensemble_submit.py --checkpoints runs/a.pt,runs/b.pt \
        --out submissions/sub_ensemble.csv [--submit -m "msg"]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, data, fusion, io  # noqa: E402

COMP = "the-freuid-challenge-2026-ijcai-ecai"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", required=True, help="comma-separated .pt paths")
    p.add_argument("--out", default="submissions/sub_ensemble.csv")
    p.add_argument("--fill", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--submit", action="store_true")
    p.add_argument("-m", "--message", default="ensemble + TTA")
    return p.parse_args()


def public_test_frame():
    root = config.PUBLIC_TEST_DIR
    rows = [{config.ID_COL: p.stem, "abs_path": str(p), config.LABEL_COL: 0}
            for p in sorted(root.rglob("*")) if p.is_file()
            and p.suffix.lower() in config.IMAGE_EXTENSIONS]
    return pd.DataFrame(rows)


def predict_tta(ckpt, frame, bs, workers, dev):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    cfgd = ck["cfg"]
    img_size = cfgd.get("img_size", 384)
    model = fusion.FusionModel(cfgd.get("backbone", "tf_efficientnetv2_m.in21k_ft_in1k"),
                               use_fusion=False, pretrained=False).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
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
    ckpts = [c.strip() for c in args.checkpoints.split(",") if c.strip() and Path(c.strip()).exists()]
    print(f"[ens] {len(ckpts)} checkpoints: {ckpts}")
    frame = public_test_frame()
    print(f"[ens] public_test: {len(frame):,}")
    acc = np.zeros(len(frame), dtype=np.float64)
    for c in ckpts:
        s = predict_tta(c, frame, args.batch_size, args.workers, dev)
        acc += s
        print(f"  [ens] {c}: mean={s.mean():.3f}")
    scores = acc / len(ckpts)

    sub = io.load_sample_submission()
    pred_map = dict(zip(frame[config.ID_COL].astype(str), scores))
    n_hit = sub[config.ID_COL].astype(str).isin(pred_map).sum()
    sub[config.LABEL_COL] = sub[config.ID_COL].astype(str).map(lambda i: float(pred_map.get(i, args.fill)))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"[ens] filled {n_hit:,}/{len(sub):,}; wrote {out} ({len(sub):,} rows)")

    if args.submit:
        cmd = ["kaggle", "competitions", "submit", "-c", COMP, "-f", str(out), "-m", args.message]
        print("[ens] submitting:", " ".join(cmd))
        print(subprocess.run(cmd, capture_output=True, text=True).stdout)


if __name__ == "__main__":
    main()
