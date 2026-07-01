#!/usr/bin/env python
"""Forensic SRM noise-residual stream: feed high-pass noise residuals (not RGB)
to a CNN. Manipulation (splice/inpaint/text edits) leaves residual fingerprints
the RGB model can miss. Self-contained: trains + writes a submission CSV.

    python3 scripts/42_train_srm.py --epochs 4 --save-name srm_cnxL \
        --out submissions/sub_srm.csv
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import aux_data, config, fusion, io, metrics, validation  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True

# three classic high-pass forensic kernels (5x5)
_KV = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], np.float32) / 12.0
_H1 = np.zeros((5, 5), np.float32); _H1[2, 1] = -1; _H1[2, 2] = 1            # first order
_H2 = np.zeros((5, 5), np.float32); _H2[2, 1:4] = [1, -2, 1]                # second order


class SRMModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        srm = nn.Conv2d(1, 3, 5, padding=2, bias=False)
        w = np.stack([_KV, _H1, _H2])[:, None]   # [3,1,5,5]
        srm.weight.data = torch.from_numpy(w); srm.weight.requires_grad = False
        self.srm = srm
        self.net = fusion.FusionModel(backbone, use_fusion=False)

    def forward(self, x, ffeat=None):
        gray = x.mean(1, keepdim=True)
        r = self.srm(gray)
        r = torch.clamp(r * 8.0, -1, 1)          # scale + clamp residual
        return self.net(r, ffeat)


class SRMDataset(Dataset):
    def __init__(self, frame, size, train):
        self.paths = frame["abs_path"].astype(str).tolist()
        self.labels = frame[config.LABEL_COL].to_numpy().astype(np.float32)
        tf = [T.Resize((size, size))]
        if train:
            tf.append(T.RandomHorizontalFlip(0.5))
        tf.append(T.ToTensor())
        self.tf = T.Compose(tf)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            im = Image.open(self.paths[i]); im.load(); im = im.convert("RGB")
            x = self.tf(im)
        except Exception:
            x = torch.zeros(3, 384, 384)
        return x, self.labels[i]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="convnextv2_base.fcmae_ft_in22k_in1k")
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-aux", type=int, default=30000)
    p.add_argument("--extra-csv", default="artifacts/pseudo_ens.csv")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--save-name", default="srm_model")
    p.add_argument("--out", default="submissions/sub_srm.csv")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def resolve(f):
    o = io.attach_image_paths(f); return o[o["path_exists"]].reset_index(drop=True)


def build(args):
    pipe = validation.ValidationPipeline(rebuild=False)
    full = pd.concat([resolve(pipe.train), resolve(pipe.val), resolve(pipe.test)], ignore_index=True)
    val = full.sample(min(5000, len(full) // 10), random_state=0)
    train = full.drop(val.index).reset_index(drop=True); val = val.reset_index(drop=True)
    cols = [config.ID_COL, config.LABEL_COL, "abs_path"]; parts = [train[cols]]
    if args.max_aux:
        aux = aux_data.load_idnet_frame([config.REPO_ROOT / "data/aux/idnet2025"])
        aux = aux[aux["abs_path"].map(lambda p: Path(p).exists())]
        if len(aux) > args.max_aux:
            aux = aux.sample(args.max_aux, random_state=0)
        parts.append(aux[cols])
    if args.extra_csv and Path(args.extra_csv).exists():
        parts.append(pd.read_csv(args.extra_csv)[cols])
    train = pd.concat(parts, ignore_index=True)
    if args.smoke:
        train = train.sample(1200, random_state=0); val = val.sample(500, random_state=0); args.epochs = 1
    print(f"[srm] train={len(train):,} val={len(val):,}")
    return train, val


def predict(model, frame, size, bs, workers, dev, tta=True):
    model.eval(); outs = []
    flips = (False, True) if tta else (False,)
    order = list(frame[config.ID_COL].astype(str))
    acc = np.zeros(len(frame))
    for flip in flips:
        tf = [T.Resize((size, size))] + ([T.RandomHorizontalFlip(1.0)] if flip else []) + [T.ToTensor()]
        ds = SRMDataset(frame.assign(**{config.LABEL_COL: 0}), size, train=False)
        ds.tf = T.Compose(tf)
        ld = DataLoader(ds, batch_size=bs, num_workers=workers, pin_memory=True)
        ps = []
        with torch.no_grad():
            for x, _ in ld:
                x = x.to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ps.append(torch.sigmoid(model(x).float()).cpu())
        acc += torch.cat(ps).numpy()
    return acc / len(flips)


def main():
    args = parse_args()
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    train, val = build(args)
    tl = DataLoader(SRMDataset(train, args.img_size, True), batch_size=args.batch_size, shuffle=True,
                    drop_last=True, num_workers=args.workers, pin_memory=True, persistent_workers=True)
    model = SRMModel(args.backbone).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(tl))
    from torchvision.ops import sigmoid_focal_loss
    best = {"audet": math.inf}
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); run = 0.0
        for bi, (x, y) in enumerate(tl):
            x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x); loss = sigmoid_focal_loss(logit, y, alpha=0.25, gamma=2.0, reduction="mean")
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            run += loss.item()
            if bi % 100 == 0:
                print(f"  ep{ep} it{bi}/{len(tl)} loss={run/(bi+1):.4f} "
                      f"({(bi+1)*args.batch_size/(time.time()-t0):.0f} img/s)", flush=True)
        sc = predict(model, val, args.img_size, args.batch_size, args.workers, dev, tta=False)
        m = metrics.freuid_score(val[config.LABEL_COL].to_numpy(), sc)
        print(f"  [ep{ep}] val FREUID={m.freuid:.4f} AuDET={m.audet:.4f}")
        if m.audet < best["audet"]:
            best = {"epoch": ep, "audet": m.audet, "freuid": m.freuid}
            torch.save({"model": model.state_dict(), "backbone": args.backbone, "img_size": args.img_size},
                       config.RUNS_DIR / f"{args.save_name}.pt")
    print(f"[srm] best={best}")

    # submission
    root = config.PUBLIC_TEST_DIR
    pub = pd.DataFrame([{config.ID_COL: p.stem, "abs_path": str(p)} for p in sorted(root.rglob("*"))
                        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS])
    ck = torch.load(config.RUNS_DIR / f"{args.save_name}.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    scores = predict(model, pub, args.img_size, args.batch_size, args.workers, dev, tta=True)
    sub = io.load_sample_submission()
    pm = dict(zip(pub[config.ID_COL].astype(str), scores))
    sub[config.LABEL_COL] = sub[config.ID_COL].astype(str).map(lambda i: float(pm.get(i, 0.5)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"[srm] wrote {args.out}")


if __name__ == "__main__":
    main()
