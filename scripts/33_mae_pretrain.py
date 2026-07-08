#!/usr/bin/env python
"""Domain masked-autoencoder (MAE-lite) pretraining on FREUID images.

Learns features specific to these documents/capture by reconstructing masked
patches of FREUID train + public_test images (unlabeled). Saves the ENCODER
weights with a 'backbone.' prefix so scripts/30 --init-from loads them into
FusionModel.backbone and finetunes on top.

    python3 scripts/33_mae_pretrain.py --backbone convnextv2_large.fcmae_ft_in22k_in1k \
        --epochs 3 --batch-size 32 --out runs/mae_convnextv2L.pt
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, data, io, validation  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="convnextv2_large.fcmae_ft_in22k_in1k")
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--patch", type=int, default=32, help="mask patch size (px)")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--out", default="runs/mae_convnextv2L.pt")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


class ImgDS(Dataset):
    def __init__(self, paths, size):
        self.paths = paths
        self.tf = data.build_transforms(data.DataConfig(img_size=size, train=False))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            im = Image.open(self.paths[i]); im.load(); im = im.convert("RGB")
            return self.tf(im)
        except Exception:
            return torch.zeros(3, 384, 384)


class Decoder(nn.Module):
    """Generic upsampler: [B,C,h,w] -> [B,3,H,W] via stride-2 transpose convs."""
    def __init__(self, cin, steps):
        super().__init__()
        layers = []; c = cin
        for i in range(steps):
            cout = max(c // 2, 64)
            layers += [nn.ConvTranspose2d(c, cout, 4, stride=2, padding=1), nn.GELU()]
            c = cout
        layers += [nn.Conv2d(c, 3, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def collect_paths():
    pipe = validation.ValidationPipeline(rebuild=False)
    df = io.attach_image_paths(pipe.table) if hasattr(pipe, "table") else None
    paths = []
    # FREUID labeled train images
    for fr in (pipe.train, pipe.val, pipe.test):
        f = io.attach_image_paths(fr)
        paths += f[f["path_exists"]]["abs_path"].astype(str).tolist()
    # public_test (target domain, unlabeled)
    root = config.PUBLIC_TEST_DIR
    paths += [str(p) for p in root.rglob("*") if p.is_file()
              and p.suffix.lower() in config.IMAGE_EXTENSIONS]
    return sorted(set(paths))


def main():
    args = parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.benchmark = True

    paths = collect_paths()
    if args.smoke:
        paths = paths[:400]; args.epochs = 1
    print(f"[mae] {len(paths):,} domain images | backbone={args.backbone}")
    loader = DataLoader(ImgDS(paths, args.img_size), batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=args.workers, pin_memory=True, persistent_workers=True)

    encoder = timm.create_model(args.backbone, pretrained=True, num_classes=0).to(dev)
    # infer feature-map shape
    with torch.no_grad():
        fmap = encoder.forward_features(torch.zeros(1, 3, args.img_size, args.img_size, device=dev))
    if fmap.dim() != 4:
        sys.exit(f"[mae] backbone forward_features not spatial ({tuple(fmap.shape)}); use a CNN backbone")
    C, h = fmap.shape[1], fmap.shape[2]
    steps = int(round(math.log2(args.img_size / h)))
    print(f"[mae] feature map C={C} h={h} -> decoder steps={steps}")
    decoder = Decoder(C, steps).to(dev)

    opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()),
                            lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader))
    g = args.img_size // args.patch  # mask grid

    for ep in range(args.epochs):
        encoder.train(); decoder.train(); t0 = time.time(); run = 0.0
        for bi, x in enumerate(loader):
            x = x.to(dev, non_blocking=True)
            B = x.size(0)
            # random patch mask (1 = masked/hidden)
            m = (torch.rand(B, 1, g, g, device=dev) < args.mask_ratio).float()
            mask = torch.nn.functional.interpolate(m, size=(args.img_size, args.img_size))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                feat = encoder.forward_features(x * (1 - mask))   # hide masked patches
                rec = decoder(feat)
                if rec.shape[-1] != x.shape[-1]:
                    rec = torch.nn.functional.interpolate(rec, size=x.shape[-2:])
                loss = (((rec - x) ** 2) * mask).sum() / (mask.sum() * 3 + 1e-6)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            run += loss.item()
            if bi % 100 == 0:
                print(f"  ep{ep} it{bi}/{len(loader)} recon_mse={run/(bi+1):.4f} "
                      f"({(bi+1)*B/(time.time()-t0):.0f} img/s)", flush=True)
        # save encoder with backbone. prefix after each epoch
        sd = {f"backbone.{k}": v for k, v in encoder.state_dict().items()}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": sd, "backbone": args.backbone, "epoch": ep}, args.out)
        print(f"  [save] {args.out} (ep{ep}, recon_mse={run/len(loader):.4f})")
    print(f"[done] MAE pretrain -> {args.out}")


if __name__ == "__main__":
    main()
