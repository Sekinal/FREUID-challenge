#!/usr/bin/env python
"""Fusion probe: do CLIP (global semantic) and region-noise (local forensic) stack?

They are orthogonal per type (CLIP wins GUINEA, region-noise wins the unseen ID).
This concatenates frozen-CLIP features (loaded from clip_features.npz, scripts/19)
with re-extracted region-noise inconsistency features (scripts/21 logic) for the
SAME images (both runs use seed 42 / sample 24000 -> identical sampling/order) and
runs leave-one-type-out, comparing CLIP, region, and CLIP+region.

    uv run scripts/22_fusion_probe.py
"""
from __future__ import annotations

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
SEED, SAMPLE = 42, 24000
CROP, TILE = 1024, 32
GRID = CROP // TILE


# ---- region-noise extraction (mirrors scripts/21) ----
class CropDataset(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]); img.load(); img = img.convert("L")
            w, h = img.size; s = min(CROP, w, h)
            img = img.crop(((w - s)//2, (h - s)//2, (w - s)//2 + s, (h - s)//2 + s))
            a = np.asarray(img, np.float32); out = np.zeros((CROP, CROP), np.float32)
            out[:a.shape[0], :a.shape[1]] = a
            return torch.from_numpy(out), True
        except Exception:
            return torch.zeros(CROP, CROP), False


def gk(dev):
    ax = torch.arange(-4, 5, dtype=torch.float32); k = torch.exp(-(ax**2)/2); k /= k.sum()
    return torch.outer(k, k).view(1, 1, 9, 9).to(dev)


def selfref(maps):
    b = maps.shape[0]; flat = maps.view(b, -1)
    med = flat.median(1, keepdim=True).values
    mad = (flat - med).abs().median(1, keepdim=True).values + 1e-6
    z = (flat - med) / mad; srt = flat.sort(1, descending=True).values
    glob = torch.stack([z.max(1).values, z.topk(5, 1).values.mean(1),
                        flat.std(1)/(flat.mean(1).abs()+1e-6),
                        srt[:, 0]-flat.median(1).values,
                        torch.quantile(flat, 0.99, 1)-torch.quantile(flat, 0.5, 1)], 1)
    m = maps.unsqueeze(1)
    nb = (F.avg_pool2d(F.pad(m, (1, 1, 1, 1), mode="replicate"), 3, 1)*9 - m)/8
    d = (m-nb).squeeze(1).view(b, -1).abs()
    nbr = torch.stack([d.max(1).values, d.topk(5, 1).values.mean(1), d.std(1),
                       torch.quantile(d, 0.99, 1)], 1)
    return torch.cat([glob, nbr], 1)


def extract_region(paths, dev):
    loader = DataLoader(CropDataset(paths), batch_size=128, num_workers=12, pin_memory=True)
    g = gk(dev); lapk = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32,
                                     device=dev).view(1, 1, 3, 3)
    feats, keep, pos, t0 = [], np.zeros(len(paths), bool), 0, time.time()
    with torch.no_grad():
        for batch, ok in loader:
            b = batch.shape[0]; x = batch.to(dev, non_blocking=True).unsqueeze(1)
            resid = x - F.conv2d(F.pad(x, (4, 4, 4, 4), mode="reflect"), g)
            lap = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), lapk)
            def ts(t, red):
                tl = t.view(b, GRID, TILE, GRID, TILE).permute(0, 1, 3, 2, 4).reshape(b, GRID, GRID, -1)
                return red(tl)
            vm = torch.log1p(ts(resid.squeeze(1), lambda v: v.var(-1)))
            mm = ts(resid.squeeze(1).abs(), lambda v: v.mean(-1))
            lm = torch.log1p(ts(lap.squeeze(1).abs(), lambda v: v.mean(-1)))
            feats.append(torch.cat([selfref(vm), selfref(mm), selfref(lm)], 1).cpu().numpy())
            keep[pos:pos+b] = ok.numpy(); pos += b
            if pos % 4096 < 128: print(f"  region {pos}/{len(paths)} ({pos/(time.time()-t0):.0f}/s)")
    X = np.concatenate(feats, 0).astype(np.float32)
    return np.where(np.isfinite(X), X, 0.0), keep


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = io.load_labels(); df = df[df["path_exists"]].reset_index(drop=True)
    cells = df.groupby([config.TYPE_COL, config.LABEL_COL])
    per = max(1, SAMPLE // cells.ngroups)
    df = pd.concat([g.sample(min(len(g), per), random_state=SEED) for _, g in cells]).reset_index(drop=True)
    labels = df[config.LABEL_COL].to_numpy().astype(np.int64)
    types = df[config.TYPE_COL].to_numpy().astype(str)
    uniq = sorted(set(types)); print(f"[fusion] {len(df):,} images")

    clip = np.load(config.ARTIFACTS_DIR / "clip_features.npz", allow_pickle=True)
    Xc = clip["X"].astype(np.float32)
    assert np.array_equal(clip["labels"].astype(np.int64), labels), "CLIP npz misaligned!"
    print(f"[fusion] CLIP features {Xc.shape} aligned OK")
    Xr, rkeep = extract_region(df["abs_path"].astype(str).tolist(), dev)
    print(f"[fusion] region features {Xr.shape}")

    valid = clip["keep"].astype(bool) & rkeep
    Xc, Xr, labels, types = Xc[valid], Xr[valid], labels[valid], types[valid]
    print(f"[fusion] valid rows: {valid.sum():,}")

    def loto(X, name):
        frs, aucs, per_t = [], [], {}
        for t in uniq:
            tr = types != t; te = ~tr
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), labels[tr])
            p = clf.predict_proba(sc.transform(X[te]))[:, 1]
            au = roc_auc_score(labels[te], p); fr = metrics.freuid_score(labels[te], p).freuid
            aucs.append(au); frs.append(fr); per_t[t] = round(float(au), 3)
        a, fr = float(np.mean(aucs)), float(np.mean(frs))
        print(f"[fusion] {name:<24} mean AUC={a:.3f} FREUID={fr:.4f} per-type={per_t}")
        return {"mean_auc": a, "mean_freuid": fr, "per_type": per_t}

    print("\n[fusion] === leave-one-type-out (linear probe) ===")
    res = {"clip": loto(Xc, "CLIP (768)"),
           "region": loto(Xr, "region-noise (27)"),
           "clip+region": loto(np.concatenate([Xc, Xr], 1), "CLIP + region (795)")}
    io.save_json("fusion_probe.json", {"n": len(df), "loto": res,
                 "refs": {"clip": 0.686, "region": 0.657, "spectral": 0.646}})
    print("[fusion] wrote artifacts/fusion_probe.json")


if __name__ == "__main__":
    main()
