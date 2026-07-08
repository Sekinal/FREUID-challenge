#!/usr/bin/env python
"""High-res intra-image noise inconsistency — the right substrate for the local idea.

The local-CLIP probe (scripts/20) showed the self-referential *idea* is sound but
CLIP's coarse semantic patches are the wrong substrate. This applies the same
"compare regions within the image" framing to **noise residuals at native
resolution**, where a spliced/inpainted/edited region leaves a different noise &
compression fingerprint than the rest of the document. Self-referential ⇒ the
type-specific baseline cancels ⇒ type-agnostic by construction.

Per image (1024 native center crop, grayscale, GPU):
- residual = gray - gaussian(gray)  (+ Laplacian high-pass)
- tile into 32x32 grid; per-tile noise energy (log variance) and mean-abs.
- self-referential descriptors of the per-tile maps: how much does the most
  outlier tile deviate from the image's own distribution (robust z, p99-p50,
  spread) AND from its spatial neighbours (NPR-on-tiles).
- LOTO linear probe vs the bars: scalar forensics 0.452, spectral 0.646, CLIP 0.686.

    uv run scripts/21_region_noise_inconsistency.py --sample 24000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
CROP = 1024          # native center crop (no resize)
TILE = 32            # -> 32x32 grid of tiles
G = CROP // TILE     # 32
LOTO_CAP = 24000


class CropDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]); img.load(); img = img.convert("L")
            w, h = img.size
            s = min(CROP, w, h)
            img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
            a = np.asarray(img, dtype=np.float32)
            out = np.zeros((CROP, CROP), np.float32)
            out[:a.shape[0], :a.shape[1]] = a
            return torch.from_numpy(out), True
        except Exception:
            return torch.zeros(CROP, CROP, dtype=torch.float32), False


def gauss_kernel(sigma=1.0, r=4, device="cuda"):
    ax = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-(ax ** 2) / (2 * sigma ** 2)); k /= k.sum()
    return torch.outer(k, k).view(1, 1, 2 * r + 1, 2 * r + 1).to(device)


def selfref(maps):
    """maps: [B, G, G] per-tile statistic -> self-referential descriptors."""
    b = maps.shape[0]
    flat = maps.view(b, -1)                                   # [B, G*G]
    med = flat.median(1, keepdim=True).values
    mad = (flat - med).abs().median(1, keepdim=True).values + 1e-6
    z = (flat - med) / mad                                    # robust z per tile
    srt = flat.sort(1, descending=True).values
    glob = torch.stack([
        z.max(1).values,                                      # most-outlier tile (robust z)
        z.topk(5, 1).values.mean(1),                          # top-5 outliers
        flat.std(1) / (flat.mean(1).abs() + 1e-6),            # coeff of variation
        srt[:, 0] - flat.median(1).values,                    # max - median
        torch.quantile(flat, 0.99, 1) - torch.quantile(flat, 0.5, 1),
    ], 1)
    # NPR-on-tiles: each tile vs 8-neighbour mean
    m = maps.unsqueeze(1)
    nb = (F.avg_pool2d(F.pad(m, (1, 1, 1, 1), mode="replicate"), 3, 1) * 9 - m) / 8
    d = (m - nb).squeeze(1).view(b, -1).abs()
    nbr = torch.stack([d.max(1).values, d.topk(5, 1).values.mean(1),
                       d.std(1), torch.quantile(d, 0.99, 1)], 1)
    return torch.cat([glob, nbr], 1)                          # [B, 9]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=24000)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[region] device={dev} CROP={CROP} grid={G}x{G}")
    df = io.load_labels(); df = df[df["path_exists"]].reset_index(drop=True)
    if args.sample:
        cells = df.groupby([config.TYPE_COL, config.LABEL_COL])
        per = max(1, args.sample // max(1, cells.ngroups))
        df = pd.concat([g.sample(min(len(g), per), random_state=args.seed)
                        for _, g in cells]).reset_index(drop=True)
    uniq = sorted(df[config.TYPE_COL].unique().tolist())
    types = df[config.TYPE_COL].to_numpy().astype(str)
    labels = df[config.LABEL_COL].to_numpy().astype(np.int64)
    print(f"[region] {len(df):,} images, types={uniq}")

    loader = DataLoader(CropDataset(df["abs_path"].astype(str).tolist()),
                        batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=(dev == "cuda"))
    gk = gauss_kernel(device=dev)
    feats = []  # per-image feature vectors
    keep = np.zeros(len(df), bool)
    pos = 0; t0 = time.time()
    with torch.no_grad():
        for batch, ok in loader:
            b = batch.shape[0]
            x = batch.to(dev, non_blocking=True).unsqueeze(1)
            resid = x - F.conv2d(F.pad(x, (4, 4, 4, 4), mode="reflect"), gk)
            lap = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"),
                           torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                        dtype=torch.float32, device=dev).view(1, 1, 3, 3))
            # per-tile maps [B, G, G]
            def tile_stat(t, reduce):
                tl = t.view(b, G, TILE, G, TILE).permute(0, 1, 3, 2, 4).reshape(b, G, G, -1)
                return reduce(tl)
            var_map = torch.log1p(tile_stat(resid.squeeze(1), lambda v: v.var(-1)))
            mabs_map = tile_stat(resid.squeeze(1).abs(), lambda v: v.mean(-1))
            lap_map = torch.log1p(tile_stat(lap.squeeze(1).abs(), lambda v: v.mean(-1)))
            fv = torch.cat([selfref(var_map), selfref(mabs_map), selfref(lap_map)], 1)
            feats.append(fv.cpu().numpy())
            keep[pos:pos + b] = ok.numpy()
            pos += b
            if pos % 4096 < args.batch_size:
                print(f"  {pos:,}/{len(df):,} ({pos/(time.time()-t0):.0f}/s)")
    X = np.concatenate(feats, 0).astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0)
    print(f"[region] features {X.shape} in {time.time()-t0:.0f}s")

    def loto(name):
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
        print(f"[region] {name}: mean AUC={a:.3f} FREUID={fr:.4f} "
              f"per-type={[round(per[t]['auc'],2) for t in uniq]}")
        return {"mean_auc": a, "mean_freuid": fr, "per_type": per}

    print("\n[region] === leave-one-type-out (linear probe) ===")
    res = loto("region noise inconsistency (27)")
    io.save_json("region_noise.json", {"n": int(keep.sum()), "n_feat": int(X.shape[1]),
                 "loto": res, "refs": {"scalar": 0.452, "spectral": 0.646, "clip": 0.686}})
    print("[region] wrote artifacts/region_noise.json")


if __name__ == "__main__":
    main()
