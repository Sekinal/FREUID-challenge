#!/usr/bin/env python
"""Predict public_test with a trained fusion model and build/submit a Kaggle CSV.

Loads a FusionModel checkpoint (from scripts/24), extracts CLIP+region features for
the 7,821 public_test images, predicts fraud scores, and writes a submission where
public_test ids get real scores and the (ignored) private dummies get a constant
fill. Optionally uploads via the Kaggle CLI.

    uv run scripts/25_predict_fusion.py --checkpoint runs/fusion_fusion_all.pt --submit
    uv run scripts/25_predict_fusion.py --checkpoint runs/fusion_nofusion_all.pt --fill 0.5
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, data, fusion, io  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--fill", type=float, default=0.5, help="constant for ignored private ids")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--tta", action="store_true", help="average original + horizontal-flip predictions")
    p.add_argument("--message", default="")
    return p.parse_args()


def public_test_frame():
    root = config.PUBLIC_TEST_DIR
    rows = [{config.ID_COL: p.stem, "abs_path": str(p), config.LABEL_COL: 0}
            for p in sorted(root.rglob("*")) if p.is_file()
            and p.suffix.lower() in config.IMAGE_EXTENSIONS]
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    cfgd = ckpt["cfg"]
    use_fusion = cfgd.get("use_fusion", True)
    backbone = cfgd.get("backbone", "tf_efficientnetv2_m.in21k_ft_in1k")
    img_size = cfgd.get("img_size", 384)
    print(f"[predict] {args.checkpoint}  backbone={backbone} use_fusion={use_fusion}")

    model = fusion.FusionModel(backbone, use_fusion=use_fusion, pretrained=False).to(dev)
    model.load_state_dict(ckpt["model"]); model.eval()

    frame = public_test_frame()
    print(f"[predict] public_test images: {len(frame):,}")
    feats, keep = fusion.extract_fusion_features(frame["abs_path"].astype(str).tolist(),
                                                 batch_size=args.batch_size, workers=args.workers)

    import torchvision.transforms as T
    tf_eval = data.build_transforms(data.DataConfig(img_size=img_size, train=False))
    loader = DataLoader(fusion.FusionDataset(frame, feats, tf_eval),
                        batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    scores = fusion._predict(model, loader, dev)
    if args.tta:
        tf_flip = T.Compose([T.Resize((img_size, img_size)), T.RandomHorizontalFlip(p=1.0),
                             T.ToTensor(), T.Normalize(data.IMAGENET_MEAN, data.IMAGENET_STD)])
        loader2 = DataLoader(fusion.FusionDataset(frame, feats, tf_flip),
                             batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
        scores = (scores + fusion._predict(model, loader2, dev)) / 2.0
        print("[predict] TTA: averaged original + hflip")
    print(f"[predict] scores: mean={scores.mean():.3f} min={scores.min():.3f} max={scores.max():.3f}")

    sub = io.load_sample_submission()
    pred_map = dict(zip(frame[config.ID_COL].astype(str), scores))
    n_hit = sub[config.ID_COL].astype(str).isin(pred_map).sum()
    sub[config.LABEL_COL] = sub[config.ID_COL].astype(str).map(
        lambda i: float(pred_map.get(i, args.fill)))
    print(f"[predict] filled {n_hit:,}/{len(sub):,} public ids; rest -> {args.fill}")

    tag = Path(args.checkpoint).stem
    out = Path(args.out) if args.out else config.SUBMISSIONS_DIR / f"sub_{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"[predict] wrote {out} ({len(sub):,} rows)")

    if args.submit:
        msg = args.message or f"fusion model {tag} (public_test scored, private fill {args.fill})"
        cmd = ["kaggle", "competitions", "submit", "-c", config.COMPETITION,
               "-f", str(out), "-m", msg]
        print("[predict] submitting:", " ".join(cmd))
        print(subprocess.run(cmd, capture_output=True, text=True).stdout)


if __name__ == "__main__":
    main()
