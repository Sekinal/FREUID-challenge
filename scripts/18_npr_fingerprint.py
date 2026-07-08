#!/usr/bin/env python
"""NPR (Neighboring Pixel Relationships, Tan et al., CVPR 2024) OOD probe.

NPR captures the *up-sampling* artifact every CNN/diffusion generator leaves: for
each l x l grid (l=2), the residual of each pixel relative to the grid's anchor
pixel. It is source-invariant (a property of the generator, not the content), so
the hypothesis is that it transfers across UNSEEN document types where scalar
forensics (LOTO AUC 0.452) and normalised spectra (0.63-0.65) only partly did.

NPR(x) = x - nearest_upsample( x[::2, ::2], 2 )      # l=2, anchor = top-left

Crucially computed on NATIVE pixels (center crop, no resize — resizing would
overwrite the generator's up-sampling fingerprint). GPU-batched on the A100.
Features per image: NPR magnitude stats + the NPR power-spectrum (radial +
downsampled, per-image normalised). Evaluated leave-one-type-out with FREUID.

    uv run scripts/18_npr_fingerprint.py                 # full dataset
    uv run scripts/18_npr_fingerprint.py --sample 20000
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

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True

CROP = 256       # native center crop (NO resize)
N_RADIAL = 64
DOWN = 32
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
            left, top = (w - s) // 2, (h - s) // 2
            img = img.crop((left, top, left + s, top + s))
            a = np.asarray(img, dtype=np.float32)
            if a.shape != (CROP, CROP):          # pad small images
                p = np.zeros((CROP, CROP), np.float32)
                p[:a.shape[0], :a.shape[1]] = a
                a = p
            return torch.from_numpy(a), True
        except Exception:
            return torch.zeros(CROP, CROP, dtype=torch.float32), False


def radial_index(side, nbins, device):
    c = side / 2.0
    y, x = torch.meshgrid(torch.arange(side, dtype=torch.float32),
                          torch.arange(side, dtype=torch.float32), indexing="ij")
    r = torch.sqrt((y - c) ** 2 + (x - c) ** 2)
    return torch.clamp((r / (r.max() + 1e-9) * nbins).long(), 0, nbins - 1).view(-1).to(device)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[npr] device={dev}")
    df = io.load_labels()
    df = df[df["path_exists"]].reset_index(drop=True)
    if args.sample:
        cells = df.groupby([config.TYPE_COL, config.LABEL_COL])
        per = max(1, args.sample // max(1, cells.ngroups))
        df = pd.concat([g.sample(min(len(g), per), random_state=args.seed)
                        for _, g in cells]).reset_index(drop=True)
    uniq = sorted(df[config.TYPE_COL].unique().tolist())
    types = df[config.TYPE_COL].to_numpy().astype(str)
    labels = df[config.LABEL_COL].to_numpy().astype(np.int64)
    print(f"[npr] {len(df):,} images, CROP={CROP}, types={uniq}")

    loader = DataLoader(CropDataset(df["abs_path"].astype(str).tolist()),
                        batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=(dev == "cuda"))
    ridx = radial_index(CROP, N_RADIAL, dev)
    rcounts = torch.zeros(N_RADIAL, device=dev).index_add_(
        0, ridx, torch.ones_like(ridx, dtype=torch.float32))

    NF_STATS = 3
    stats = np.zeros((len(df), NF_STATS), np.float32)
    rad = np.zeros((len(df), N_RADIAL), np.float32)
    down = np.zeros((len(df), DOWN * DOWN), np.float32)
    keep = np.zeros(len(df), bool)
    pos = 0; t0 = time.time()
    with torch.no_grad():
        for batch, ok in loader:
            b = batch.shape[0]
            x = batch.to(dev, non_blocking=True).unsqueeze(1)            # [b,1,H,W]
            anchor = F.interpolate(x[:, :, ::2, ::2], scale_factor=2, mode="nearest")
            npr = x - anchor                                            # NPR map
            # magnitude stats (per image)
            flat = npr.view(b, -1)
            mu = flat.mean(1); sd = flat.std(1) + 1e-9
            kurt = (((flat - mu[:, None]) / sd[:, None]) ** 4).mean(1) - 3.0
            mabs = flat.abs().mean(1)
            stats[pos:pos + b] = torch.stack([sd, kurt, mabs], 1).cpu().numpy()
            # NPR power spectrum (per-image normalised shape)
            Fc = torch.fft.fftshift(torch.fft.fft2(npr), dim=(-2, -1))
            power = torch.log1p(Fc.abs() ** 2).squeeze(1)
            pmu = power.mean((-2, -1), keepdim=True); psd = power.std((-2, -1), keepdim=True) + 1e-9
            power = (power - pmu) / psd
            rp = torch.zeros(b, N_RADIAL, device=dev).index_add_(1, ridx, power.view(b, -1)) / rcounts
            dn = F.avg_pool2d(power.unsqueeze(1), CROP // DOWN).view(b, -1)
            rad[pos:pos + b] = rp.cpu().numpy()
            down[pos:pos + b] = dn.cpu().numpy()
            keep[pos:pos + b] = ok.numpy()
            pos += b
            if pos % 8192 < args.batch_size:
                print(f"  {pos:,}/{len(df):,} ({pos/(time.time()-t0):.0f}/s)")
    print(f"[npr] computed in {time.time()-t0:.0f}s")

    def loto(X, name):
        m = keep.copy()
        if m.sum() > LOTO_CAP:
            rng = np.random.default_rng(args.seed)
            sel = rng.choice(np.where(m)[0], LOTO_CAP, replace=False)
            m = np.zeros_like(m); m[sel] = True
        Xk, yk, tk = X[m], labels[m], types[m]
        frs, aucs = [], []
        for t in uniq:
            tr = tk != t; te = ~tr
            if len(np.unique(yk[tr])) < 2 or len(np.unique(yk[te])) < 2:
                continue
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                 l2_regularization=1.0, random_state=args.seed)
            clf.fit(Xk[tr], yk[tr])
            p = clf.predict_proba(Xk[te])[:, 1]
            frs.append(metrics.freuid_score(yk[te], p).freuid)
            aucs.append(roc_auc_score(yk[te], p))
        a, fr = float(np.mean(aucs)), float(np.mean(frs))
        print(f"[npr] LOTO ({name}): mean AUC={a:.3f}  mean FREUID={fr:.4f}")
        return a, fr

    s_auc, s_fr = loto(stats, "NPR stats(3)")
    r_auc, r_fr = loto(rad, f"NPR spectrum radial {N_RADIAL}")
    d_auc, d_fr = loto(down, f"NPR spectrum downsampled {DOWN}x{DOWN}")
    c_auc, c_fr = loto(np.concatenate([stats, rad, down], 1), "NPR all (stats+radial+down)")

    io.save_json("npr_fingerprint.json", {
        "n": int(keep.sum()), "crop": CROP,
        "loto": {"stats": {"auc": s_auc, "freuid": s_fr},
                 "radial": {"auc": r_auc, "freuid": r_fr},
                 "downsampled": {"auc": d_auc, "freuid": d_fr},
                 "combined": {"auc": c_auc, "freuid": c_fr}},
        "refs": {"scalar_forensics_auc": 0.452, "spectral_best_auc": 0.646},
    })
    print("[npr] wrote artifacts/npr_fingerprint.json")


if __name__ == "__main__":
    main()
