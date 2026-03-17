"""PyTorch Lightning DataModule for crop classification training.

Reads manifest CSVs produced by ``prepare_dataset.py``, loads ``.npz``
chips, and yields batches in the format expected by TerraTorch's
``ClassificationTask``:  ``{"image": Tensor, "label": Tensor}``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


class CropChipDataset(Dataset):
    """Dataset that loads .npz chips and returns image + label dicts."""

    def __init__(
        self,
        manifest_path: str | Path,
        augmentations: dict[str, Any] | None = None,
    ) -> None:
        self.df = pd.read_csv(manifest_path)
        self.augmentations = augmentations or {}
        log.info("Dataset: %d samples from %s", len(self.df), manifest_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        data = np.load(row["chip_path"], allow_pickle=True)
        pixels = data["pixels"].astype(np.float32) / 10_000.0

        image = torch.from_numpy(pixels)
        label = torch.tensor(int(row["crop_class_id"]), dtype=torch.long)

        # Augmentations
        if self.augmentations.get("random_horizontal_flip", False):
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=[-1])

        if self.augmentations.get("random_vertical_flip", False):
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=[-2])

        band_dropout = self.augmentations.get("band_dropout", 0.0)
        if band_dropout > 0:
            mask = torch.rand(image.shape[0]) > band_dropout
            image = image * mask.view(-1, 1, 1).float()

        return {"image": image, "label": label}


class CropDataModule(L.LightningDataModule):
    """Lightning DataModule wrapping manifest-based crop chip datasets."""

    def __init__(
        self,
        dataset_dir: str | Path,
        batch_size: int = 32,
        num_workers: int = 4,
        augmentations: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentations = augmentations

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self.train_ds = CropChipDataset(
                self.dataset_dir / "train_manifest.csv",
                augmentations=self.augmentations,
            )
            self.val_ds = CropChipDataset(
                self.dataset_dir / "val_manifest.csv",
                augmentations=None,
            )
        if stage in ("test", None):
            self.test_ds = CropChipDataset(
                self.dataset_dir / "test_manifest.csv",
                augmentations=None,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
