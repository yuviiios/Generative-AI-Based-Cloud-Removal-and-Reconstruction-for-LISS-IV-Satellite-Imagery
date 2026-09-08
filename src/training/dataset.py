"""
PyTorch Dataset for cloud removal training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Tuple, Optional
from src.preprocessing.augmentation import SatelliteAugmentor


class CloudRemovalDataset(Dataset):
    """
    Loads (cloudy, clear, mask) triplets from .npy files.
    """

    def __init__(
        self,
        data_dir: str,
        augment: bool = False,
        split: str = "train",
    ):
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.augmentor = SatelliteAugmentor(p=0.5) if augment else None

        clear_dir  = self.data_dir / "clear"
        cloudy_dir = self.data_dir / "cloudy"
        mask_dir   = self.data_dir / "masks"

        self.clear_files  = sorted(clear_dir.glob("*.npy"))
        self.cloudy_files = sorted(cloudy_dir.glob("*.npy"))
        self.mask_files   = sorted(mask_dir.glob("*.npy"))

        assert len(self.clear_files) == len(self.cloudy_files) == len(self.mask_files), \
            "Mismatch in number of clear/cloudy/mask files"

    def __len__(self) -> int:
        return len(self.clear_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clear  = np.load(self.clear_files[idx]).astype(np.float32)   # (H, W, C)
        cloudy = np.load(self.cloudy_files[idx]).astype(np.float32)
        mask   = np.load(self.mask_files[idx]).astype(np.float32)    # (H, W)

        if self.augmentor is not None:
            cloudy, clear, mask = self.augmentor(cloudy, clear, mask)

        # Convert to (C, H, W) tensors
        cloudy_t = torch.from_numpy(cloudy.transpose(2, 0, 1))
        clear_t  = torch.from_numpy(clear.transpose(2, 0, 1))
        mask_t   = torch.from_numpy(mask).unsqueeze(0)   # (1, H, W)

        return cloudy_t, clear_t, mask_t


def build_dataloaders(
    data_dir: str,
    train_split: float = 0.7,
    val_split: float = 0.15,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 42,
):
    """Create train/val/test DataLoaders."""
    full_ds = CloudRemovalDataset(data_dir, augment=False)
    n = len(full_ds)
    n_train = int(n * train_split)
    n_val   = int(n * val_split)
    n_test  = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )
    train_ds.dataset.augment = True
    train_ds.dataset.augmentor = SatelliteAugmentor(p=0.5)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"[Dataset] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
