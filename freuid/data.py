"""PyTorch dataset and transforms for identity-document images.

Training transforms include **capture-simulating augmentation** (JPEG recompression,
downscale-upscale, perspective, blur, lighting jitter). The private test set
emphasises *captured / print-and-capture* images while our training data is 99.97%
digital, so simulating capture degradation is the main lever for the digital->captured
domain shift we cannot otherwise validate (see agents_docs/03).
"""
from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from . import config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DataConfig:
    img_size: int = 384
    train: bool = True
    capture_aug: bool = True


# --------------------------------------------------------------------------
# Capture-simulating PIL transforms (module-level so they pickle for workers)
# --------------------------------------------------------------------------
class RandomJPEG:
    """Re-encode through JPEG at a random quality -> simulates recompression."""

    def __init__(self, quality_range: tuple[int, int] = (35, 95), p: float = 0.5) -> None:
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=random.randint(*self.quality_range))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img


class RandomDownscale:
    """Downscale then upscale back -> simulates resolution loss / soft capture."""

    def __init__(self, scale_range: tuple[float, float] = (0.4, 1.0), p: float = 0.4) -> None:
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            w, h = img.size
            s = random.uniform(*self.scale_range)
            small = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
            img = small.resize((w, h), Image.BILINEAR)
        return img


def build_transforms(cfg: DataConfig) -> T.Compose:
    resize = T.Resize((cfg.img_size, cfg.img_size))
    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if not cfg.train:
        return T.Compose([resize, T.ToTensor(), normalize])

    if not cfg.capture_aug:
        return T.Compose([resize, T.RandomHorizontalFlip(0.5), T.ToTensor(), normalize])

    return T.Compose([
        resize,
        T.RandomHorizontalFlip(0.5),
        T.RandomPerspective(distortion_scale=0.2, p=0.3),          # capture angle
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # lighting
        T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.3),   # focus/capture blur
        RandomDownscale(scale_range=(0.4, 1.0), p=0.4),                # resolution loss
        RandomJPEG(quality_range=(35, 95), p=0.5),                     # recompression
        T.ToTensor(),
        normalize,
    ])


class DocumentDataset(Dataset):
    """Labeled document images referenced by ``abs_path`` in the frame."""

    def __init__(
        self,
        frame: pd.DataFrame,
        cfg: DataConfig | None = None,
        path_col: str = "abs_path",
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.cfg = cfg or DataConfig()
        self.path_col = path_col
        self.transform = build_transforms(self.cfg)
        if self.path_col not in self.frame.columns:
            raise ValueError(f"Missing column {self.path_col}")
        if config.LABEL_COL not in self.frame.columns and self.cfg.train:
            raise ValueError(f"Missing label column {config.LABEL_COL}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        path = Path(str(row[self.path_col]))
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        if config.LABEL_COL in self.frame.columns:
            label = float(row[config.LABEL_COL])
            return image, torch.tensor(label, dtype=torch.float32), str(row[config.ID_COL])
        return image, str(row[config.ID_COL])
