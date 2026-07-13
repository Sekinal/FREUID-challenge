#!/usr/bin/env python3
"""FREUID Challenge 2026 - standalone inference (Docker entrypoint).

Reads a flat directory of images, scores them with one or more checkpoints
(prob-averaged, optional hflip TTA), writes CSV id,label.

Single pass over the data: each decoded batch is scored by every model and
TTA view before the next batch is loaded, so decode cost is amortized.

    python3 infer.py --data-dir /data --out /submissions/submission.csv \
        --checkpoints w/a.pt,w/b.pt,w/c.pt --batch-size 32 --workers 8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Small /dev/shm inside sandboxed containers breaks the default worker
# sharing strategy; file_system avoids any shm dependency.
torch.multiprocessing.set_sharing_strategy("file_system")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXTS = {".jpeg", ".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class Net(nn.Module):
    """Backbone + MLP head; mirrors freuid.fusion.FusionModel with use_fusion=False."""

    def __init__(self, backbone_name: str):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        bd = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(bd), nn.Dropout(0.2),
            nn.Linear(bd, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, img):
        return self.head(self.backbone(img)).squeeze(1)


class ImgDataset(Dataset):
    def __init__(self, paths, img_size):
        self.paths = paths
        self.tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), i


def load_model(path, dev):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    backbone = cfg.get("backbone", "tf_efficientnetv2_m.in21k_ft_in1k")
    img_size = cfg.get("img_size", 1024)
    model = Net(backbone)
    sd = ckpt.get("model", ckpt)
    # strip fusion keys if present (use_fusion=False checkpoints have none)
    sd = {k: v for k, v in sd.items() if not k.startswith("fnorm")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    assert all(m.startswith("fnorm") for m in missing), f"missing keys: {missing[:5]}"
    model.eval().to(dev, memory_format=torch.channels_last)
    return model, img_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/data")
    ap.add_argument("--out", default="/submissions/submission.csv")
    ap.add_argument("--checkpoints", required=True, help="comma-separated .pt paths")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-tta", action="store_true", help="disable hflip TTA")
    ap.add_argument("--limit", type=int, default=0, help="benchmark: only first N images")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    paths = sorted(p for p in Path(args.data_dir).iterdir() if p.suffix.lower() in EXTS)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        sys.exit(f"[infer] no images found in {args.data_dir}")
    print(f"[infer] {len(paths):,} images | device={dev}", flush=True)

    models, sizes = [], set()
    for cp in args.checkpoints.split(","):
        m, s = load_model(cp.strip(), dev)
        models.append(m)
        sizes.add(s)
        print(f"[infer] loaded {cp.strip()} (img_size={s})", flush=True)
    assert len(sizes) == 1, f"mixed img sizes: {sizes}"
    img_size = sizes.pop()
    n_views = len(models) * (1 if args.no_tta else 2)
    print(f"[infer] img_size={img_size} | {len(models)} models x "
          f"{'1 view' if args.no_tta else '2 views (hflip TTA)'} = {n_views} forwards/img", flush=True)

    loader = DataLoader(ImgDataset(paths, img_size), batch_size=args.batch_size,
                        num_workers=args.workers, pin_memory=True)
    scores = np.zeros(len(paths), np.float64)
    t0 = time.time()
    done = 0
    with torch.no_grad():
        for x, idx in loader:
            x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            views = [x] if args.no_tta else [x, torch.flip(x, dims=[3])]
            acc = torch.zeros(x.shape[0], device=dev, dtype=torch.float32)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                for v in views:
                    for m in models:
                        acc += torch.sigmoid(m(v).float())
            scores[idx.numpy()] = (acc / n_views).cpu().numpy()
            done += x.shape[0]
            if done % (args.batch_size * 20) < args.batch_size:
                el = time.time() - t0
                print(f"  {done:,}/{len(paths):,}  {done/el:.1f} img/s  "
                      f"eta {(len(paths)-done)/max(done/el,1e-9)/60:.1f} min", flush=True)

    el = time.time() - t0
    print(f"[infer] done in {el/60:.2f} min ({len(paths)/el:.1f} img/s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": [p.stem for p in paths], "label": scores}).to_csv(out, index=False)
    print(f"[infer] wrote {out} ({len(paths):,} rows)", flush=True)


if __name__ == "__main__":
    main()
