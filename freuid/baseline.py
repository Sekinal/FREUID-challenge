"""Baseline fraud detector: pretrained timm backbone + binary head."""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config, data, io, metrics


@dataclass
class BaselineConfig:
    model_name: str = "efficientnet_b2"
    img_size: int = 384
    batch_size: int = 384
    epochs: int = 3
    lr: float = 2e-4
    weight_decay: float = 1e-4
    num_workers: int = 8
    seed: int = 42
    amp: bool = True
    max_train_samples: int | None = None
    aug: str = "none"            # "none" | "domain"
    loss_type: str = "bce"       # "bce" | "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


_VIT_LIKE = ("vit", "dinov2", "deit", "beit", "eva", "samvit")


def build_model(model_name: str, pretrained: bool = True, img_size: int | None = None) -> nn.Module:
    kwargs: dict = {}
    # ViT-family backbones need img_size to (re)interpolate position embeddings
    # (e.g. DINOv2 is pretrained at 518px); CNNs don't accept the kwarg.
    if img_size is not None and any(k in model_name for k in _VIT_LIKE):
        kwargs["img_size"] = img_size
    return timm.create_model(model_name, pretrained=pretrained, num_classes=1, **kwargs)


def _maybe_subset(frame: pd.DataFrame, max_samples: int | None) -> pd.DataFrame:
    if max_samples is None or len(frame) <= max_samples:
        return frame
    return frame.sample(n=max_samples, random_state=0).reset_index(drop=True)


def _make_loader(frame: pd.DataFrame, cfg: BaselineConfig, train: bool) -> DataLoader:
    ds = data.DocumentDataset(
        frame,
        cfg=data.DataConfig(img_size=cfg.img_size, train=train, aug=cfg.aug if train else "none"),
    )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train and len(ds) > cfg.batch_size,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )


@torch.no_grad()
def predict_frame(
    model: nn.Module,
    frame: pd.DataFrame,
    cfg: BaselineConfig,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    model.eval()
    loader = _make_loader(frame, cfg, train=False)
    scores: list[float] = []
    ids: list[str] = []
    for batch in loader:
        if len(batch) == 3:
            images, _, batch_ids = batch
        else:
            images, batch_ids = batch
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits)
        scores.extend(probs.detach().cpu().numpy().tolist())
        ids.extend(list(batch_ids))
    return np.asarray(scores, dtype=np.float64), ids


def evaluate_frame(
    model: nn.Module,
    frame: pd.DataFrame,
    cfg: BaselineConfig,
    device: torch.device,
) -> metrics.MetricResult:
    scores, ids = predict_frame(model, frame, cfg, device)
    id_to_label = dict(zip(frame[config.ID_COL].astype(str), frame[config.LABEL_COL]))
    y_true = np.asarray([id_to_label[i] for i in ids], dtype=np.int8)
    return metrics.freuid_score(y_true, scores)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    cfg: BaselineConfig,
    train: bool,
) -> float:
    model.train(train)
    total_loss = 0.0
    n_batches = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels, _ in tqdm(loader, leave=False, desc="train" if train else "eval"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
                logits = model(images).squeeze(1)
                loss = loss_fn(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
    return total_loss / max(n_batches, 1)


def pos_weight_from_frame(frame: pd.DataFrame) -> float:
    y = frame[config.LABEL_COL].to_numpy()
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    return n_neg / n_pos


def train_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: BaselineConfig | None = None,
    run_dir: Path | None = None,
    test_df: pd.DataFrame | None = None,
    prefiltered: bool = False,
) -> dict:
    """Train a simple EfficientNet baseline and score val/test with FREUID.

    When ``prefiltered`` is True the train frame is assumed to already carry a
    valid ``abs_path`` column (e.g. it mixes auxiliary images that do not live
    in the FREUID image index), so train paths are not re-resolved — only rows
    whose ``abs_path`` exists on disk are kept. Val/test are always FREUID
    frames, re-resolved against the FREUID index for honest cross-country scores.
    """
    cfg = cfg or BaselineConfig()
    run_dir = Path(run_dir) if run_dir else config.RUNS_DIR / "baseline"
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = get_device()

    if prefiltered:
        train_df = _maybe_subset(train_df, cfg.max_train_samples).copy()
        keep = train_df["abs_path"].map(lambda p: bool(p) and Path(str(p)).exists())
        train_df = train_df[keep].reset_index(drop=True)
    else:
        train_df = io.filter_with_images(_maybe_subset(train_df, cfg.max_train_samples))
    val_df = io.filter_with_images(val_df)
    if test_df is not None:
        test_df = io.filter_with_images(test_df)

    train_loader = _make_loader(train_df, cfg, train=True)
    val_loader = _make_loader(val_df, cfg, train=False)

    model = build_model(cfg.model_name, pretrained=True, img_size=cfg.img_size).to(device)
    pos_weight = torch.tensor([pos_weight_from_frame(train_df)], device=device)
    if cfg.loss_type == "focal":
        from torchvision.ops import sigmoid_focal_loss

        def loss_fn(logits, labels):
            return sigmoid_focal_loss(
                logits, labels, alpha=cfg.focal_alpha, gamma=cfg.focal_gamma, reduction="mean"
            )
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1))

    history: list[dict] = []
    best_freuid = math.inf
    best_path = run_dir / "best.pt"

    for epoch in range(1, cfg.epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, loss_fn, device, cfg, train=True)
        val_loss = _run_epoch(model, val_loader, optimizer, loss_fn, device, cfg, train=False)
        val_metrics = evaluate_frame(model, val_df, cfg, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_freuid": val_metrics.freuid,
            "val_audet": val_metrics.audet,
            "val_apcer_at_1pct_bpcer": val_metrics.apcer_at_1pct_bpcer,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": epoch}, run_dir / "last.pt")
        if val_metrics.freuid < best_freuid:
            best_freuid = val_metrics.freuid
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": epoch}, best_path)

    # reload best checkpoint for final scoring
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

    val_final = evaluate_frame(model, val_df, cfg, device)
    result = {
        "config": asdict(cfg),
        "device": str(device),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "pos_weight": float(pos_weight.item()),
        "history": history,
        "val": {
            "freuid": val_final.freuid,
            "audet": val_final.audet,
            "apcer_at_1pct_bpcer": val_final.apcer_at_1pct_bpcer,
            "eer": val_final.eer,
        },
        "checkpoint_best": str(best_path),
        "checkpoint_last": str(run_dir / "last.pt"),
    }
    if test_df is not None and not test_df.empty:
        test_final = evaluate_frame(model, test_df, cfg, device)
        result["test"] = {
            "freuid": test_final.freuid,
            "audet": test_final.audet,
            "apcer_at_1pct_bpcer": test_final.apcer_at_1pct_bpcer,
            "eer": test_final.eer,
            "n": len(test_df),
        }
    (run_dir / "results.json").write_text(json.dumps(result, indent=2))
    io.save_json("baseline_results.json", result)
    return result


def load_model(checkpoint_path: Path, device: torch.device | None = None) -> tuple[nn.Module, BaselineConfig]:
    device = device or get_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = BaselineConfig(**ckpt["cfg"])
    model = build_model(cfg.model_name, pretrained=False, img_size=cfg.img_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg
