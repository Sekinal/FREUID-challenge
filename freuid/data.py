"""PyTorch dataset and transforms for identity-document images."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms as T

from . import config

# Some dataset images (esp. external/IDNet) are slightly truncated; tolerate
# them instead of crashing a DataLoader worker mid-epoch.
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DataConfig:
    img_size: int = 384
    train: bool = True
    aug: str = "none"          # "none" | "domain" (cross-capture augmentation)


class _AlbumentationsTransform:
    """Adapter so an albumentations pipeline can take a PIL image like torchvision."""

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    def __call__(self, image):
        return self.pipeline(image=np.asarray(image))["image"]


def _domain_transform(img_size: int) -> _AlbumentationsTransform:
    """Cross-domain augmentation: simulate capture/compression/lighting shifts.

    Targets the FREUID failure mode — a model that overfits the training
    countries' capture artifacts and fails on unseen ones. Uses albumentations.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    pipeline = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.GaussNoise(p=0.3),
            A.ImageCompression(quality_range=(55, 100), p=0.4),
            A.Perspective(scale=(0.02, 0.07), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    return _AlbumentationsTransform(pipeline)


def build_transforms(cfg: DataConfig):
    if cfg.train and cfg.aug == "domain":
        return _domain_transform(cfg.img_size)
    if cfg.train:
        return T.Compose(
            [
                T.Resize((cfg.img_size, cfg.img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return T.Compose(
        [
            T.Resize((cfg.img_size, cfg.img_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


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
