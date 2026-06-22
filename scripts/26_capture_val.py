#!/usr/bin/env python
"""Capture-shift local validation — a proxy that tracks the captured public LB.

Our in-distribution / held-out-type val saturates (~0) because it's all-digital and
type isn't the real shift; it did NOT predict the public LB (fusion looked fine
locally but lost on Kaggle). This evaluates a checkpoint on a held-out FREUID set
(the train-all val carve, 5,000 imgs, seed 0 -> identical across runs, never
trained on) with **capture degradation applied** (JPEG + downscale + blur),
simulating print-and-capture. Reports clean vs degraded FREUID.

Validate the proxy: it should (a) be non-saturated on degraded, and (b) rank the
known checkpoints correctly (no-fusion 0.187 < fusion 0.225 on public).

    uv run scripts/26_capture_val.py --checkpoint runs/fusion_nofusion_all.pt
    uv run scripts/26_capture_val.py --checkpoint runs/fusion_fusion_all.pt --jpeg 45 --scale 0.6
"""
from __future__ import annotations

import argparse
import io as _io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, data, fusion, io, metrics, validation  # noqa: E402


def make_degrade(jpeg: int, scale: float, blur: float):
    """Deterministic PIL->PIL capture degradation (same image for backbone & features)."""
    def f(img: Image.Image) -> Image.Image:
        if scale < 1.0:
            w, h = img.size
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
            img = img.resize((w, h), Image.BILINEAR)
        if blur > 0:
            img = img.filter(ImageFilter.GaussianBlur(blur))
        if jpeg < 100:
            buf = _io.BytesIO(); img.save(buf, "JPEG", quality=jpeg); buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img
    return f


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--jpeg", type=int, default=50)
    p.add_argument("--scale", type=float, default=0.65)
    p.add_argument("--blur", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=12)
    return p.parse_args()


def heldout_val(n):
    pipe = validation.ValidationPipeline(rebuild=False)
    full = pd.concat([pipe.train, pipe.val, pipe.test], ignore_index=True)
    val = full.sample(min(5000, max(1, len(full) // 10)), random_state=0)  # == train-all carve
    val = io.attach_image_paths(val.head(n))
    return val[val["path_exists"]].reset_index(drop=True)


def evaluate(frame, model, dev, use_fusion, degrade, args):
    paths = frame["abs_path"].astype(str).tolist()
    if use_fusion:
        feats, _ = fusion.extract_fusion_features(paths, batch_size=args.batch_size,
                                                  workers=args.workers, pre=degrade)
    else:
        feats = np.zeros((len(frame), fusion.FUSION_DIM), np.float32)
    clean = [T.Resize((384, 384)), T.ToTensor(), T.Normalize(data.IMAGENET_MEAN, data.IMAGENET_STD)]
    tf = T.Compose(([T.Lambda(degrade)] if degrade else []) + clean)
    loader = DataLoader(fusion.FusionDataset(frame, feats, tf), batch_size=args.batch_size,
                        num_workers=args.workers, pin_memory=True)
    scores = fusion._predict(model, loader, dev)
    return metrics.freuid_score(frame[config.LABEL_COL].to_numpy(), scores)


def main():
    args = parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    cfgd = ckpt["cfg"]; use_fusion = cfgd.get("use_fusion", True)
    model = fusion.FusionModel(cfgd.get("backbone", "tf_efficientnetv2_m.in21k_ft_in1k"),
                               use_fusion=use_fusion, pretrained=False).to(dev)
    model.load_state_dict(ckpt["model"]); model.eval()
    frame = heldout_val(args.n)
    print(f"[capval] {args.checkpoint}  use_fusion={use_fusion}  n={len(frame):,}  "
          f"degrade(jpeg={args.jpeg},scale={args.scale},blur={args.blur})")

    degrade = make_degrade(args.jpeg, args.scale, args.blur)
    m_clean = evaluate(frame, model, dev, use_fusion, None, args)
    m_deg = evaluate(frame, model, dev, use_fusion, degrade, args)
    print(f"[capval] CLEAN    FREUID={m_clean.freuid:.4f}  AuDET={m_clean.audet:.4f}")
    print(f"[capval] DEGRADED FREUID={m_deg.freuid:.4f}  AuDET={m_deg.audet:.4f}  "
          f"APCER@1%={m_deg.apcer_at_1pct_bpcer:.4f}")
    io.save_json(f"capval_{Path(args.checkpoint).stem}.json",
                 {"checkpoint": args.checkpoint, "use_fusion": use_fusion,
                  "clean_freuid": m_clean.freuid, "degraded_freuid": m_deg.freuid,
                  "degrade": {"jpeg": args.jpeg, "scale": args.scale, "blur": args.blur}})


if __name__ == "__main__":
    main()
