#!/usr/bin/env python
"""Fourier-domain fraud fingerprint analysis — hunting a *type-agnostic* OOD lever.

GPU-accelerated (A100): CPU DataLoader workers decode+resize, the GPU does the
Gaussian residual, 2-D FFT, log-power, per-image normalisation, radial profile,
downsampling, and per-(type,label) accumulation in batches. Fast enough to run
the FULL dataset, which denoises the mean spectra so the cross-type correlation
is trustworthy.

Scalar forensic features (scripts/14-16) collapse on unseen types because they
encode type-specific absolute levels. This probes a different hypothesis: image
*generation/manipulation* leaves a content-independent periodic signature in the
frequency spectrum. Decisive question:

    Is the (fraud - genuine) mean spectral difference CONSISTENT across types?
    High cross-type correlation => universal generation fingerprint == OOD lever.

Also runs leave-one-type-out classifiers on the spectral representation (radial
profile + downsampled 2-D spectrum) vs the scalar-forensics OOD AUC of 0.452.

    uv run scripts/17_spectral_fingerprint.py                 # full dataset
    uv run scripts/17_spectral_fingerprint.py --sample 20000
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

S = 256          # resize side (uniform -> spectra comparable across images)
N_RADIAL = 64    # radial-profile bins
DOWN = 32        # downsampled 2-D spectrum side for the LOTO classifier
LOTO_CAP = 24000  # cap rows fed to the GBM (keeps it fast); spectra use all


class ImgDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]); img.load()
            img = img.convert("L").resize((S, S), Image.BILINEAR)
            return torch.from_numpy(np.asarray(img, dtype=np.float32)), True
        except Exception:
            return torch.zeros(S, S, dtype=torch.float32), False


def gaussian_kernel(sigma=1.0, radius=4, device="cuda"):
    ax = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    k2 = torch.outer(k, k)
    return k2.view(1, 1, 2 * radius + 1, 2 * radius + 1).to(device)


def radial_index(device="cuda"):
    cy = cx = S / 2.0
    y, x = torch.meshgrid(torch.arange(S, dtype=torch.float32),
                          torch.arange(S, dtype=torch.float32), indexing="ij")
    r = torch.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    idx = torch.clamp((r / (r.max() + 1e-9) * N_RADIAL).long(), 0, N_RADIAL - 1)
    return idx.view(-1).to(device)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=0, help="0 = full dataset")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[spectral] device={dev} ({torch.cuda.get_device_name(0) if dev=='cuda' else 'cpu'})")

    df = io.load_labels()
    df = df[df["path_exists"]].reset_index(drop=True)
    if args.sample:
        cells = df.groupby([config.TYPE_COL, config.LABEL_COL])
        per = max(1, args.sample // max(1, cells.ngroups))
        df = pd.concat([g.sample(min(len(g), per), random_state=args.seed)
                        for _, g in cells]).reset_index(drop=True)
    uniq = sorted(df[config.TYPE_COL].unique().tolist())
    tindex = {t: i for i, t in enumerate(uniq)}
    types = df[config.TYPE_COL].to_numpy().astype(str)
    labels = df[config.LABEL_COL].to_numpy().astype(np.int64)
    tidx = np.array([tindex[t] for t in types], dtype=np.int64)
    group_all = tidx * 2 + labels
    print(f"[spectral] {len(df):,} images, S={S}, types={uniq}")

    loader = DataLoader(ImgDataset(df["abs_path"].astype(str).tolist()),
                        batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=(dev == "cuda"))
    gk = gaussian_kernel(device=dev)
    ridx = radial_index(device=dev)
    rad_counts = torch.zeros(N_RADIAL, device=dev).index_add_(
        0, ridx, torch.ones_like(ridx, dtype=torch.float32))

    sums = torch.zeros(len(uniq) * 2, S, S, device=dev)
    cnts = torch.zeros(len(uniq) * 2, device=dev)
    rad = np.zeros((len(df), N_RADIAL), np.float32)
    down = np.zeros((len(df), DOWN * DOWN), np.float32)
    keep = np.zeros(len(df), bool)

    t0 = time.time(); pos = 0
    grp_t = torch.from_numpy(group_all).to(dev)
    with torch.no_grad():
        for batch, ok in loader:
            b = batch.shape[0]
            g = batch.to(dev, non_blocking=True).unsqueeze(1)          # [b,1,S,S]
            blur = F.conv2d(F.pad(g, (4, 4, 4, 4), mode="reflect"), gk)
            resid = g - blur
            Fc = torch.fft.fftshift(torch.fft.fft2(resid), dim=(-2, -1))
            power = torch.log1p(Fc.abs() ** 2).squeeze(1)              # [b,S,S]
            mu = power.mean(dim=(-2, -1), keepdim=True)
            sd = power.std(dim=(-2, -1), keepdim=True) + 1e-9
            power = (power - mu) / sd
            # radial profile
            flat = power.view(b, -1)
            rp = torch.zeros(b, N_RADIAL, device=dev).index_add_(
                1, ridx, flat) / rad_counts
            # downsampled spectrum
            dn = F.avg_pool2d(power.unsqueeze(1), kernel_size=S // DOWN).view(b, -1)
            # accumulate mean spectra per (type,label)
            grp = grp_t[pos:pos + b]
            sums.index_add_(0, grp, power)
            cnts.index_add_(0, grp, torch.ones(b, device=dev))

            rad[pos:pos + b] = rp.cpu().numpy()
            down[pos:pos + b] = dn.cpu().numpy()
            keep[pos:pos + b] = ok.numpy()
            pos += b
            if pos % 8192 < args.batch_size:
                print(f"  {pos:,}/{len(df):,} ({pos/(time.time()-t0):.0f}/s)")
    print(f"[spectral] spectra computed in {time.time()-t0:.0f}s")

    sums_np = sums.cpu().numpy(); cnts_np = cnts.cpu().numpy()
    means = {}
    for t, ti in tindex.items():
        for l in (0, 1):
            gi = ti * 2 + l
            if cnts_np[gi] > 0:
                means[(t, l)] = sums_np[gi] / cnts_np[gi]
    diffs = {t: means[(t, 1)] - means[(t, 0)] for t in uniq
             if (t, 1) in means and (t, 0) in means}
    keys = list(diffs)
    corr = np.eye(len(keys))
    for a in range(len(keys)):
        for b2 in range(a + 1, len(keys)):
            c = float(np.corrcoef(diffs[keys[a]].ravel(), diffs[keys[b2]].ravel())[0, 1])
            corr[a, b2] = corr[b2, a] = c
    off = corr[~np.eye(len(keys), dtype=bool)]
    mean_off = float(off.mean()) if off.size else float("nan")
    print("\n[spectral] cross-type correlation of (fraud-genuine) spectral difference:")
    print("   types:", keys)
    for a in range(len(keys)):
        print("   " + " ".join(f"{corr[a,b2]:+.2f}" for b2 in range(len(keys))))
    print(f"   --> MEAN off-diagonal correlation = {mean_off:+.3f}")
    print("       (>~0.5 == universal fraud fingerprint across types -> OOD lever;")
    print("        ~0    == type-specific, no transferable spectral signature)")

    def loto(X, name):
        m = keep.copy()
        if m.sum() > LOTO_CAP:  # cap for GBM speed, keep balance via random subsample
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
        print(f"[spectral] LOTO ({name}): mean AUC={a:.3f}  mean FREUID={fr:.4f}")
        return a, fr

    rad_auc, rad_fr = loto(rad, f"radial profile {N_RADIAL}")
    down_auc, down_fr = loto(down, f"downsampled spectrum {DOWN}x{DOWN}")

    np.savez_compressed(config.ARTIFACTS_DIR / "spectral_means.npz",
                        **{f"{t}|{l}": means[(t, l)] for (t, l) in means})
    io.save_json("spectral_fingerprint.json", {
        "n": int(keep.sum()), "S": S, "device": dev,
        "cross_type_diff_corr": corr.tolist(), "diff_types": keys,
        "mean_offdiag_corr": mean_off,
        "loto_radial": {"auc": rad_auc, "freuid": rad_fr},
        "loto_downsampled_spectrum": {"auc": down_auc, "freuid": down_fr},
        "scalar_forensics_loto_auc_ref": 0.452,
    })
    print("[spectral] wrote artifacts/spectral_fingerprint.json + spectral_means.npz")


if __name__ == "__main__":
    main()
