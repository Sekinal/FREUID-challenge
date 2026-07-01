#!/usr/bin/env python
"""Heavy multi-view TTA + RANK-AVERAGE ensembling -> submission CSV.

FREUID is a pure ranking metric (AuDET + APCER@1%BPCER depend only on score
order), so: (1) heavy TTA on the 7,821 public_test images (cheap) stabilises
each model's ranking; (2) ensembling by RANK (percentile) instead of raw
probability is the principled combiner for rank metrics (a miscalibrated model
can't dominate). CNN backbones get multi-scale TTA; window/patch ViTs (swin/vit/
dinov2) keep their fixed size (hflip only).

    python3 scripts/31_polish_submit.py --checkpoints runs/a.pt,runs/b.pt \
        --rank-avg --out submissions/sub_p_ens.csv [--submit -m msg]
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
FIXED = ("swin", "vit", "dinov2", "beit", "deit")  # fixed-input backbones


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", required=True)
    p.add_argument("--out", default="submissions/sub_polish.csv")
    p.add_argument("--rank-avg", action="store_true", help="ensemble by rank/percentile")
    p.add_argument("--scales", default="1.0,1.15", help="multi-scale factors (CNN only)")
    p.add_argument("--fill", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--submit", action="store_true")
    p.add_argument("-m", "--message", default="heavy TTA + rank-avg")
    return p.parse_args()


def public_test_frame():
    root = config.PUBLIC_TEST_DIR
    rows = [{config.ID_COL: p.stem, "abs_path": str(p), config.LABEL_COL: 0}
            for p in sorted(root.rglob("*")) if p.is_file()
            and p.suffix.lower() in config.IMAGE_EXTENSIONS]
    return pd.DataFrame(rows)


def _predict_one(model, frame, feats, size, flip, bs, workers, dev):
    tf = [T.Resize((size, size))]
    if flip:
        tf.append(T.RandomHorizontalFlip(p=1.0))
    tf += [T.ToTensor(), T.Normalize(data.IMAGENET_MEAN, data.IMAGENET_STD)]
    ld = DataLoader(fusion.FusionDataset(frame, feats, T.Compose(tf)), batch_size=bs,
                    num_workers=workers, pin_memory=True)
    return fusion._predict(model, ld, dev)


def heavy_tta(ckpt, frame, scales, bs, workers, dev):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    cfgd = ck["cfg"]
    backbone = cfgd.get("backbone", "tf_efficientnetv2_m.in21k_ft_in1k")
    img_size = cfgd.get("img_size", 384)
    model = fusion.FusionModel(backbone, use_fusion=False, pretrained=False).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    feats = np.zeros((len(frame), fusion.FUSION_DIM), np.float32)
    use_scales = [1.0] if any(k in backbone for k in FIXED) else scales
    views, n = None, 0
    for s in use_scales:
        size = int(round(img_size * s / 32) * 32) if s != 1.0 else img_size
        for flip in (False, True):
            p = _predict_one(model, frame, feats, size, flip, bs, workers, dev)
            views = p if views is None else views + p
            n += 1
    print(f"  [tta] {ckpt}: {n} views (scales={use_scales})")
    return views / n


def main():
    args = parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    scales = [float(x) for x in args.scales.split(",")]
    ckpts = [c.strip() for c in args.checkpoints.split(",") if c.strip() and Path(c.strip()).exists()]
    print(f"[polish] {len(ckpts)} ckpts | rank-avg={args.rank_avg} | scales={scales}")
    frame = public_test_frame()

    per_model = [heavy_tta(c, frame, scales, args.batch_size, args.workers, dev) for c in ckpts]
    if args.rank_avg:
        # average percentile ranks (robust for a rank-based metric)
        def pct(a):
            return np.argsort(np.argsort(a)) / max(len(a) - 1, 1)
        scores = np.mean([pct(p) for p in per_model], axis=0)
        print("[polish] combined by RANK-average")
    else:
        scores = np.mean(per_model, axis=0)
        print("[polish] combined by probability-average")

    sub = io.load_sample_submission()
    pred = dict(zip(frame[config.ID_COL].astype(str), scores))
    n_hit = sub[config.ID_COL].astype(str).isin(pred).sum()
    sub[config.LABEL_COL] = sub[config.ID_COL].astype(str).map(lambda i: float(pred.get(i, args.fill)))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"[polish] filled {n_hit:,}/{len(sub):,}; wrote {out}")

    if args.submit:
        cmd = ["kaggle", "competitions", "submit", "-c", COMP, "-f", str(out), "-m", args.message]
        print(subprocess.run(cmd, capture_output=True, text=True).stdout)


if __name__ == "__main__":
    main()
