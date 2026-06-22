"""Fusion model: EffNetV2 backbone + frozen-CLIP + region-noise streams.

Probe analysis (agents_docs/04) found two orthogonal *transferable* signals —
frozen CLIP (global semantic, LOTO AUC 0.686) and self-referential region-noise
inconsistency (local forensic, 0.94 on the unseen ID type) — but naive frozen-
feature concat (0.667) could not exploit their complementarity. The fix is a
*trained* head that can gate per example: backbone image features (in-type
discriminative power) concatenated with the two precomputed frozen streams,
trained with IDNet aux on the leakage-safe LOTO harness.

This module provides:
- ``extract_fusion_features`` — CLIP(768) + region-noise(27) per image (one decode).
- ``FusionModel`` — timm backbone + MLP head over [backbone | fusion]; ``use_fusion``
  toggles the streams for a clean ablation.
- ``FusionDataset`` / ``train_fusion`` — training loop (focal/bce, AMP, FREUID eval).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

from . import config, data, io, metrics

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLIP_MODEL = "vit_base_patch16_clip_224.openai"
CLIP_DIM, REGION_DIM = 768, 27
FUSION_DIM = CLIP_DIM + REGION_DIM          # 795
RCROP, RTILE = 1024, 32
RGRID = RCROP // RTILE


# ==========================================================================
# Frozen feature extraction (CLIP + region-noise), mirrors scripts/19 & 21
# ==========================================================================
class _ExtractDS(Dataset):
    def __init__(self, paths, clip_tf):
        self.paths, self.clip_tf = paths, clip_tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]); img.load(); rgb = img.convert("RGB")
            clip_t = self.clip_tf(rgb)
            g = rgb.convert("L"); w, h = g.size; s = min(RCROP, w, h)
            g = g.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
            a = np.asarray(g, np.float32)
            out = np.zeros((RCROP, RCROP), np.float32); out[:a.shape[0], :a.shape[1]] = a
            return clip_t, torch.from_numpy(out), True
        except Exception:
            return torch.zeros(3, 224, 224), torch.zeros(RCROP, RCROP), False


def _gauss(dev):
    ax = torch.arange(-4, 5, dtype=torch.float32); k = torch.exp(-(ax ** 2) / 2); k /= k.sum()
    return torch.outer(k, k).view(1, 1, 9, 9).to(dev)


def _selfref(maps):
    b = maps.shape[0]; flat = maps.view(b, -1)
    med = flat.median(1, keepdim=True).values
    mad = (flat - med).abs().median(1, keepdim=True).values + 1e-6
    z = (flat - med) / mad; srt = flat.sort(1, descending=True).values
    glob = torch.stack([z.max(1).values, z.topk(5, 1).values.mean(1),
                        flat.std(1) / (flat.mean(1).abs() + 1e-6),
                        srt[:, 0] - flat.median(1).values,
                        torch.quantile(flat, 0.99, 1) - torch.quantile(flat, 0.5, 1)], 1)
    m = maps.unsqueeze(1)
    nb = (F.avg_pool2d(F.pad(m, (1, 1, 1, 1), mode="replicate"), 3, 1) * 9 - m) / 8
    d = (m - nb).squeeze(1).view(b, -1).abs()
    nbr = torch.stack([d.max(1).values, d.topk(5, 1).values.mean(1), d.std(1),
                       torch.quantile(d, 0.99, 1)], 1)
    return torch.cat([glob, nbr], 1)


def _region_batch(x, gk, lapk):
    b = x.shape[0]
    resid = x - F.conv2d(F.pad(x, (4, 4, 4, 4), mode="reflect"), gk)
    lap = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), lapk)

    def ts(t, red):
        tl = t.view(b, RGRID, RTILE, RGRID, RTILE).permute(0, 1, 3, 2, 4).reshape(b, RGRID, RGRID, -1)
        return red(tl)
    vm = torch.log1p(ts(resid.squeeze(1), lambda v: v.var(-1)))
    mm = ts(resid.squeeze(1).abs(), lambda v: v.mean(-1))
    lm = torch.log1p(ts(lap.squeeze(1).abs(), lambda v: v.mean(-1)))
    return torch.cat([_selfref(vm), _selfref(mm), _selfref(lm)], 1)


def extract_fusion_features(paths, device=None, batch_size=128, workers=12, log_every=8192):
    """Return (X[n, 795] float32, keep[n] bool) = [CLIP768 | region27] per path."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    clip = timm.create_model(CLIP_MODEL, pretrained=True, num_classes=0).eval().to(dev)
    for p in clip.parameters():
        p.requires_grad_(False)
    cfg = timm.data.resolve_model_data_config(clip)
    tf = timm.data.create_transform(**cfg, is_training=False)
    loader = DataLoader(_ExtractDS(list(paths), tf), batch_size=batch_size,
                        num_workers=workers, pin_memory=(dev == "cuda"))
    gk = _gauss(dev)
    lapk = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32,
                        device=dev).view(1, 1, 3, 3)
    n = len(paths)
    X = np.zeros((n, FUSION_DIM), np.float32); keep = np.zeros(n, bool)
    pos = 0; t0 = time.time()
    with torch.no_grad():
        for clip_t, reg_t, ok in loader:
            b = clip_t.shape[0]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                cf = clip(clip_t.to(dev, non_blocking=True)).float()
            rf = _region_batch(reg_t.to(dev, non_blocking=True).unsqueeze(1), gk, lapk)
            X[pos:pos + b, :CLIP_DIM] = cf.cpu().numpy()
            X[pos:pos + b, CLIP_DIM:] = rf.cpu().numpy()
            keep[pos:pos + b] = ok.numpy(); pos += b
            if pos % log_every < batch_size:
                print(f"  [feat] {pos:,}/{n:,} ({pos/(time.time()-t0):.0f}/s)")
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X, keep


# ==========================================================================
# Fusion model
# ==========================================================================
class FusionModel(nn.Module):
    def __init__(self, backbone_name="tf_efficientnetv2_m.in21k_ft_in1k",
                 use_fusion=True, fusion_dim=FUSION_DIM, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        bd = self.backbone.num_features
        self.use_fusion = use_fusion
        # normalise the (precomputed) fusion stream before mixing with learned feats
        self.fnorm = nn.LayerNorm(fusion_dim) if use_fusion else None
        in_dim = bd + (fusion_dim if use_fusion else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Dropout(0.2),
            nn.Linear(in_dim, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, img, ffeat=None):
        f = self.backbone(img)
        if self.use_fusion:
            f = torch.cat([f, self.fnorm(ffeat)], dim=1)
        return self.head(f).squeeze(1)


# ==========================================================================
# Training
# ==========================================================================
@dataclass
class FusionConfig:
    backbone: str = "tf_efficientnetv2_m.in21k_ft_in1k"
    img_size: int = 384
    epochs: int = 3
    batch_size: int = 96
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 12
    use_fusion: bool = True
    loss_type: str = "focal"          # "focal" | "bce"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    amp: bool = True
    capture_aug: bool = True
    max_train: int = 0                 # 0 = all


class FusionDataset(Dataset):
    def __init__(self, frame, feats, transform):
        self.paths = frame["abs_path"].astype(str).tolist()
        self.labels = frame[config.LABEL_COL].to_numpy().astype(np.float32)
        self.feats = feats               # [n, FUSION_DIM] aligned with frame rows
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]); img.load(); img = img.convert("RGB")
            x = self.transform(img)
        except Exception:
            x = torch.zeros(3, 384, 384)
        return x, torch.from_numpy(self.feats[i]), self.labels[i]


def _loss_fn(cfg):
    if cfg.loss_type == "focal":
        from torchvision.ops import sigmoid_focal_loss

        def f(logits, y):
            return sigmoid_focal_loss(logits, y, alpha=cfg.focal_alpha,
                                      gamma=cfg.focal_gamma, reduction="mean")
        return f
    return nn.BCEWithLogitsLoss()


@torch.no_grad()
def _predict(model, loader, dev):
    model.eval(); out = []
    for x, ff, _ in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logit = model(x.to(dev, non_blocking=True), ff.to(dev, non_blocking=True))
        out.append(torch.sigmoid(logit.float()).cpu().numpy())
    return np.concatenate(out)


def train_fusion(train_df, train_feats, eval_sets, cfg: FusionConfig, save_name=None):
    """Train fusion model. eval_sets: {name: (frame, feats)}. Returns metrics dict."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if cfg.max_train and len(train_df) > cfg.max_train:
        idx = np.random.default_rng(0).choice(len(train_df), cfg.max_train, replace=False)
        train_df = train_df.iloc[idx].reset_index(drop=True); train_feats = train_feats[idx]

    tf_train = data.build_transforms(data.DataConfig(img_size=cfg.img_size, train=True,
                                                     capture_aug=cfg.capture_aug))
    tf_eval = data.build_transforms(data.DataConfig(img_size=cfg.img_size, train=False))
    tr_loader = DataLoader(FusionDataset(train_df, train_feats, tf_train),
                           batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                           num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True)
    eval_loaders = {k: DataLoader(FusionDataset(fr, fe, tf_eval), batch_size=cfg.batch_size,
                                  num_workers=cfg.num_workers, pin_memory=True)
                    for k, (fr, fe) in eval_sets.items()}

    model = FusionModel(cfg.backbone, use_fusion=cfg.use_fusion).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * len(tr_loader))
    loss_fn = _loss_fn(cfg)

    best = {"freuid": math.inf}
    for ep in range(cfg.epochs):
        model.train(); t0 = time.time(); run = 0.0
        for bi, (x, ff, y) in enumerate(tr_loader):
            x = x.to(dev, non_blocking=True); ff = ff.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                logit = model(x, ff); loss = loss_fn(logit, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            run += loss.item()
            if bi % 50 == 0:
                print(f"  ep{ep} it{bi}/{len(tr_loader)} loss={run/(bi+1):.4f} "
                      f"({(bi+1)*cfg.batch_size/(time.time()-t0):.0f} img/s)")
        # eval
        ep_metrics = {}
        for name, (fr, fe) in eval_sets.items():
            scores = _predict(model, eval_loaders[name], dev)
            m = metrics.freuid_score(fr[config.LABEL_COL].to_numpy(), scores)
            ep_metrics[name] = {"freuid": m.freuid, "audet": m.audet,
                                "apcer_at_1pct_bpcer": m.apcer_at_1pct_bpcer}
            print(f"  [ep{ep}] {name}: FREUID={m.freuid:.4f} AuDET={m.audet:.4f} "
                  f"APCER@1%={m.apcer_at_1pct_bpcer:.4f}")
        # checkpoint by the first eval set's AuDET (smooth)
        key = next(iter(eval_sets))
        if ep_metrics[key]["audet"] < best.get("audet", math.inf):
            best = {"epoch": ep, "audet": ep_metrics[key]["audet"], "metrics": ep_metrics}
            if save_name:
                config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "cfg": cfg.__dict__},
                           config.RUNS_DIR / f"{save_name}.pt")
    return {"best": best, "use_fusion": cfg.use_fusion}
