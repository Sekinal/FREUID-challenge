"""PyTorch dataset and transforms for identity-document images."""
from __future__ import annotations

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


def build_transforms(cfg: DataConfig) -> T.Compose:
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
