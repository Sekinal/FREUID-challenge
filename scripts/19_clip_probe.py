#!/usr/bin/env python
"""CLIP frozen-feature OOD probe (UnivFD; Ojha et al.) for FREUID.

UnivFD's finding: a *frozen* CLIP-ViT image encoder clusters real images, and
fakes fall outside that distribution — a simple linear probe on these features
generalises to UNSEEN generators far better than fine-tuned CNNs (which overfit
seen ones). We test the analogous claim for unseen *document types*: do frozen
CLIP features + a linear probe transfer across held-out types better than the
low-level transforms (scalar forensics 0.452, normalised spectra 0.65, NPR ~0.5)?

Frozen ViT-L/14 (OpenAI CLIP) features on the A100, then leave-one-type-out with
a linear probe (UnivFD-faithful) and a GBM, scored with FREUID. Must stay frozen
(fine-tuning causes catastrophic forgetting of the transferable prior).

    uv run scripts/19_clip_probe.py                 # full dataset
    uv run scripts/19_clip_probe.py --sample 20000
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
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, io, metrics  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True
MODEL = "vit_base_patch16_clip_224.openai"
LOTO_CAP = 30000


class ImgDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]); img.load(); img = img.convert("RGB")
            return self.transform(img), True
        except Exception:
            return torch.zeros(3, 224, 224), False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[clip] device={dev}  model={MODEL}")
    model = timm.create_model(MODEL, pretrained=True, num_classes=0).eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**cfg, is_training=False)
    feat_dim = model.num_features
    print(f"[clip] frozen, feature dim = {feat_dim}")

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
    print(f"[clip] {len(df):,} images, types={uniq}")

    loader = DataLoader(ImgDataset(df["abs_path"].astype(str).tolist(), transform),
                        batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=(dev == "cuda"))
    X = np.zeros((len(df), feat_dim), np.float32)
    keep = np.zeros(len(df), bool)
    pos = 0; t0 = time.time()
    with torch.no_grad():
        for batch, ok in loader:
            b = batch.shape[0]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                f = model(batch.to(dev, non_blocking=True))
            X[pos:pos + b] = f.float().cpu().numpy()
            keep[pos:pos + b] = ok.numpy()
            pos += b
            if pos % 8192 < args.batch_size:
                print(f"  {pos:,}/{len(df):,} ({pos/(time.time()-t0):.0f}/s)")
    print(f"[clip] features extracted in {time.time()-t0:.0f}s")

    np.savez_compressed(config.ARTIFACTS_DIR / "clip_features.npz",
                        X=X, labels=labels, types=types, keep=keep)

    def loto(make_clf, name):
        m = keep.copy()
        if m.sum() > LOTO_CAP:
            rng = np.random.default_rng(args.seed)
            sel = rng.choice(np.where(m)[0], LOTO_CAP, replace=False)
            m = np.zeros_like(m); m[sel] = True
        Xk, yk, tk = X[m], labels[m], types[m]
        frs, aucs, per = [], [], {}
        for t in uniq:
            tr = tk != t; te = ~tr
            if len(np.unique(yk[tr])) < 2 or len(np.unique(yk[te])) < 2:
                continue
            sc = StandardScaler().fit(Xk[tr])
            clf = make_clf().fit(sc.transform(Xk[tr]), yk[tr])
            p = clf.predict_proba(sc.transform(Xk[te]))[:, 1]
            fr = metrics.freuid_score(yk[te], p).freuid
            au = roc_auc_score(yk[te], p)
            frs.append(fr); aucs.append(au)
            per[t] = {"auc": float(au), "freuid": float(fr)}
            print(f"   hold-out {t:<16} AUC={au:.3f}  FREUID={fr:.4f}")
        a, fr = float(np.mean(aucs)), float(np.mean(frs))
        print(f"[clip] LOTO ({name}): mean AUC={a:.3f}  mean FREUID={fr:.4f}")
        return {"mean_auc": a, "mean_freuid": fr, "per_type": per}

    print("\n[clip] === linear probe (UnivFD-faithful) ===")
    lin = loto(lambda: LogisticRegression(max_iter=2000, C=1.0), "logreg")
    print("\n[clip] === GBM probe ===")
    gbm = loto(lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                      l2_regularization=1.0, random_state=args.seed),
               "histgb")

    io.save_json("clip_probe.json", {
        "n": int(keep.sum()), "model": MODEL, "feat_dim": int(feat_dim),
        "loto_linear": lin, "loto_gbm": gbm,
        "refs": {"scalar_forensics_auc": 0.452, "spectral_best_auc": 0.646,
                 "npr_auc": 0.53},
    })
    print("[clip] wrote artifacts/clip_probe.json")


if __name__ == "__main__":
    main()
