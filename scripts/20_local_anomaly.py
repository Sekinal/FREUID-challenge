#!/usr/bin/env python
"""Local / region-level anomaly probe: "NPR on CLIP patch tokens".

Synthesis of two ideas: FREUID fraud is *localized* (swapped photo, edited field,
altered MRZ), and CLIP features transfer across types. Global CLIP (scripts/19,
LOTO AUC 0.686) dilutes a local tamper across the whole embedding. Here we keep it
*local and self-referential* — compare regions WITHIN each image, so the
type-specific baseline cancels out (a type-agnostic tamper signal):

- CLIP ViT-B/16 -> 196 patch tokens (14x14 regions), frozen.
- intra-image anomaly: each patch's cosine distance to the image's mean patch
  (is one region an outlier vs the whole document?).
- CLIP-NPR: each patch's distance to its 4 spatial neighbours' mean
  (is a region inconsistent with its *surroundings*? — NPR's neighbouring
  relationship, applied to semantic patches).
- summarise both as a few scalars (max / top-k / spread) -> LOTO probe, compared
  to the global-CLIP 0.686. Also fuse global + local.

    uv run scripts/20_local_anomaly.py --sample 24000
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
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, io, metrics  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True
MODEL = "vit_base_patch16_clip_224.openai"
GRID = 14            # 224/16
LOTO_CAP = 24000


class ImgDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]); img.load(); img = img.convert("RGB")
            return self.transform(img), True
        except Exception:
            return torch.zeros(3, 224, 224), False


def _stats(d):  # d: [B, K] per-patch scalar -> summary features
    srt = torch.sort(d, dim=1, descending=True).values
    return torch.stack([
        srt[:, 0],                       # max
        srt[:, :5].mean(1),              # top-5 mean
        d.mean(1),                       # mean
        d.std(1),                        # spread
        torch.quantile(d, 0.9, dim=1),   # p90
        (srt[:, 0] - d.mean(1)) / (d.std(1) + 1e-6),  # peakiness (z of max)
    ], dim=1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=24000)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[local] device={dev} model={MODEL}")
    model = timm.create_model(MODEL, pretrained=True, num_classes=0).eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**cfg, is_training=False)
    npref = getattr(model, "num_prefix_tokens", 1)

    df = io.load_labels(); df = df[df["path_exists"]].reset_index(drop=True)
    if args.sample:
        cells = df.groupby([config.TYPE_COL, config.LABEL_COL])
        per = max(1, args.sample // max(1, cells.ngroups))
        df = pd.concat([g.sample(min(len(g), per), random_state=args.seed)
                        for _, g in cells]).reset_index(drop=True)
    uniq = sorted(df[config.TYPE_COL].unique().tolist())
    types = df[config.TYPE_COL].to_numpy().astype(str)
    labels = df[config.LABEL_COL].to_numpy().astype(np.int64)
    print(f"[local] {len(df):,} images, types={uniq}")

    loader = DataLoader(ImgDataset(df["abs_path"].astype(str).tolist(), transform),
                        batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=(dev == "cuda"))
    G = np.zeros((len(df), model.num_features), np.float32)   # global CLS
    A = np.zeros((len(df), 6), np.float32)                    # intra-image anomaly
    Nn = np.zeros((len(df), 6), np.float32)                   # CLIP-NPR (neighbour)
    keep = np.zeros(len(df), bool)
    pos = 0; t0 = time.time()
    # 4-neighbour kernel for spatial smoothing on the 14x14 patch grid
    with torch.no_grad():
        for batch, ok in loader:
            b = batch.shape[0]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                tok = model.forward_features(batch.to(dev, non_blocking=True))  # [b, N, C]
            tok = tok.float()
            cls = tok[:, 0]                                   # [b,C]
            patches = tok[:, npref:]                          # [b,196,C]
            pn = F.normalize(patches, dim=-1)
            # intra-image anomaly: distance to mean patch
            meanp = F.normalize(patches.mean(1), dim=-1)      # [b,C]
            dist_mean = 1 - (pn * meanp[:, None]).sum(-1)     # [b,196]
            # CLIP-NPR: distance to 4-neighbour mean on the 14x14 grid
            grid = pn.transpose(1, 2).reshape(b, -1, GRID, GRID)        # [b,C,14,14]
            neigh = F.avg_pool2d(F.pad(grid, (1, 1, 1, 1), mode="replicate"),
                                 kernel_size=3, stride=1) * 9.0
            neigh = (neigh - grid) / 8.0                       # mean of 8 neighbours
            neigh = F.normalize(neigh, dim=1)
            dist_nb = (1 - (grid * neigh).sum(1)).reshape(b, -1)        # [b,196]
            G[pos:pos + b] = cls.cpu().numpy()
            A[pos:pos + b] = _stats(dist_mean).cpu().numpy()
            Nn[pos:pos + b] = _stats(dist_nb).cpu().numpy()
            keep[pos:pos + b] = ok.numpy()
            pos += b
            if pos % 8192 < args.batch_size:
                print(f"  {pos:,}/{len(df):,} ({pos/(time.time()-t0):.0f}/s)")
    print(f"[local] features in {time.time()-t0:.0f}s")

    def loto(X, name):
        m = keep.copy()
        if m.sum() > LOTO_CAP:
            rng = np.random.default_rng(args.seed)
            sel = rng.choice(np.where(m)[0], LOTO_CAP, replace=False)
            m = np.zeros_like(m); m[sel] = True
        Xk, yk, tk = X[m], labels[m], types[m]
        frs, aucs, per = [], [], {}
        for t in uniq:
            tr = tk != t; te = ~tr
            sc = StandardScaler().fit(Xk[tr])
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xk[tr]), yk[tr])
            p = clf.predict_proba(sc.transform(Xk[te]))[:, 1]
            au = roc_auc_score(yk[te], p); fr = metrics.freuid_score(yk[te], p).freuid
            aucs.append(au); frs.append(fr); per[t] = {"auc": float(au), "freuid": float(fr)}
        a, fr = float(np.mean(aucs)), float(np.mean(frs))
        print(f"[local] {name:<28} mean AUC={a:.3f}  mean FREUID={fr:.4f}  "
              f"per-type AUC={[round(per[t]['auc'],2) for t in uniq]}")
        return {"mean_auc": a, "mean_freuid": fr, "per_type": per}

    print("\n[local] === leave-one-type-out (linear probe) ===")
    res = {
        "global_cls": loto(G, "global CLS (=scripts/19)"),
        "anomaly": loto(A, "intra-image anomaly (6)"),
        "clip_npr": loto(Nn, "CLIP-NPR neighbour (6)"),
        "local_all": loto(np.concatenate([A, Nn], 1), "local all (anomaly+npr, 12)"),
        "global_plus_local": loto(np.concatenate([G, A, Nn], 1), "global + local (768+12)"),
    }
    io.save_json("local_anomaly.json", {
        "n": int(keep.sum()), "model": MODEL,
        "loto": res, "refs": {"global_clip_auc": 0.686, "spectral_auc": 0.646}})
    print("[local] wrote artifacts/local_anomaly.json")


if __name__ == "__main__":
    main()
