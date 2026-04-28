"""
dataset.py
----------
Handles loading, preprocessing, augmentation, and splitting of the Sen1Floods11
dataset for binary flood segmentation using Sentinel-1 SAR imagery.

Dataset source:
  Bonafilia et al. (2020) – Sen1Floods11
  GitHub: https://github.com/cloudtostreet/Sen1Floods11
  GEE / GCS bucket: gs://sen1floods11

Expected on-disk structure after download:
  data/
    sen1floods11/
      v1.1/
        data/
          flood_events/
            HandLabeled/
              S1Hand/          <- Sentinel-1 image chips  (*_S1Hand.tif)
              LabelHand/       <- Expert label masks      (*_LabelHand.tif)
        splits/
          flood_handlabeled_split.csv   <- official train/val/test split CSV

Each .tif chip is 512×512 pixels with 2 bands: VV (band 0) and VH (band 1).
Label masks contain: 0 = non-flooded, 1 = flooded, -1 = invalid/no-data.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
# Channel statistics computed from the Sen1Floods11 training split
# (log10(σ⁰) converted to dB: 10 × log10(σ⁰))
# ---------------------------------------------------------------------------
TRAIN_MEAN_VV = -13.2   # dB
TRAIN_STD_VV  =   4.8   # dB
TRAIN_MEAN_VH = -20.1   # dB
TRAIN_STD_VH  =   5.3   # dB

# Clamp range applied before normalisation to remove extreme outlier pixels
SAR_CLAMP_MIN = -50.0   # dB
SAR_CLAMP_MAX =   5.0   # dB


def load_tif(path: str) -> np.ndarray:
    """
    Read a GeoTIFF file and return its data as a float32 NumPy array.

    Parameters
    ----------
    path : str
        Full path to the .tif file.

    Returns
    -------
    np.ndarray
        Array of shape (H, W, C) where C is the number of bands.
    """
    with rasterio.open(path) as src:
        data = src.read()                      # (C, H, W)
    return data.transpose(1, 2, 0).astype(np.float32)  # → (H, W, C)


def sar_to_db(arr: np.ndarray) -> np.ndarray:
    """
    Convert linear Sentinel-1 sigma-naught values to decibels.
    Small positive epsilon avoids log(0).

    Parameters
    ----------
    arr : np.ndarray
        Linear sigma-naught values (float32).

    Returns
    -------
    np.ndarray
        Values in dB (float32).
    """
    arr = np.clip(arr, a_min=1e-10, a_max=None)
    return 10.0 * np.log10(arr).astype(np.float32)


def normalise_sar(chip: np.ndarray) -> np.ndarray:
    """
    Apply per-channel Z-score normalisation using training-set statistics.
    Input is expected to already be in dB and clamped.

    Parameters
    ----------
    chip : np.ndarray
        SAR chip of shape (H, W, 2) in dB.

    Returns
    -------
    np.ndarray
        Normalised chip, shape (H, W, 2).
    """
    chip = chip.copy()
    chip[..., 0] = (chip[..., 0] - TRAIN_MEAN_VV) / TRAIN_STD_VV
    chip[..., 1] = (chip[..., 1] - TRAIN_MEAN_VH) / TRAIN_STD_VH
    return chip


def preprocess_sar(chip: np.ndarray, already_db: bool = False) -> np.ndarray:
    """
    Full preprocessing pipeline for a Sentinel-1 SAR chip:
      1. Convert linear → dB (if not already in dB)
      2. Clamp to [SAR_CLAMP_MIN, SAR_CLAMP_MAX]
      3. Z-score normalise with training-set statistics

    Parameters
    ----------
    chip : np.ndarray
        SAR chip of shape (H, W, 2).
    already_db : bool
        If True, skip the log transformation step.

    Returns
    -------
    np.ndarray
        Preprocessed chip, shape (H, W, 2).
    """
    if not already_db:
        chip = sar_to_db(chip)
    chip = np.clip(chip, SAR_CLAMP_MIN, SAR_CLAMP_MAX)
    chip = normalise_sar(chip)
    return chip


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = 512) -> A.Compose:
    """
    Albumentations augmentation pipeline for training.
    All transforms are applied identically to both image and mask.

    Augmentations applied:
      - Random horizontal / vertical flips
      - Random 90° rotation
      - Random crop then resize back to image_size
      - Gaussian noise injection
      - Random brightness-contrast per channel

    Parameters
    ----------
    image_size : int
        Final spatial size of chips after augmentation (default 512).

    Returns
    -------
    A.Compose
        Composed augmentation pipeline.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomCrop(height=384, width=384, p=0.5),
        A.Resize(height=image_size, width=image_size),
        A.GaussNoise(var_limit=(25.0, 225.0), p=0.3),   # σ ≈ 5–15 dB
        A.RandomBrightnessContrast(brightness_limit=0.15,
                                    contrast_limit=0.15, p=0.3),
        ToTensorV2()
    ])


def get_val_transforms(image_size: int = 512) -> A.Compose:
    """
    Minimal transform pipeline for validation and test sets
    (resize only, no stochastic augmentation).
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        ToTensorV2()
    ])


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class Sen1Floods11Dataset(Dataset):
    """
    PyTorch Dataset for the Sen1Floods11 hand-labelled flood segmentation task.

    Parameters
    ----------
    root_dir : str
        Root path of the sen1floods11 dataset (contains v1.1/).
    split : str
        One of 'train', 'val', or 'test'.
    transforms : A.Compose or None
        Albumentations transform pipeline. If None, ToTensorV2 is applied.
    image_size : int
        Spatial size of output chips.
    already_db : bool
        Set True if dataset TIFs store values already in dB.
    """

    SPLIT_CSV = os.path.join("v1.1", "splits", "flood_handlabeled_split.csv")
    IMG_DIR   = os.path.join("v1.1", "data", "flood_events",
                              "HandLabeled", "S1Hand")
    MASK_DIR  = os.path.join("v1.1", "data", "flood_events",
                              "HandLabeled", "LabelHand")

    # Mapping from CSV split column values to our split names
    SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transforms=None,
        image_size: int = 512,
        already_db: bool = False
    ):
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"

        self.root_dir   = root_dir
        self.split      = split
        self.transforms = transforms
        self.image_size = image_size
        self.already_db = already_db

        # Load the official split CSV
        csv_path = os.path.join(root_dir, self.SPLIT_CSV)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"Split CSV not found at {csv_path}.\n"
                "Please download the Sen1Floods11 dataset and place it under data/sen1floods11/."
            )

        df = pd.read_csv(csv_path)

        # The CSV has columns: [flood, S1, S2, label, split]
        # 'split' values: 'train', 'valid', 'test'
        # Map 'valid' → 'val' to match our naming convention
        target_split = "valid" if split == "val" else split
        subset = df[df["split"] == target_split].reset_index(drop=True)

        self.image_paths = [
            os.path.join(root_dir, self.IMG_DIR, fname)
            for fname in subset["S1"].values
        ]
        self.mask_paths = [
            os.path.join(root_dir, self.MASK_DIR, fname)
            for fname in subset["label"].values
        ]

        # Fall back to ToTensorV2 if no transforms provided
        if self.transforms is None:
            self.transforms = A.Compose([
                A.Resize(height=image_size, width=image_size),
                ToTensorV2()
            ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        image : torch.Tensor  shape (2, H, W) float32
        mask  : torch.Tensor  shape (1, H, W) float32  values in {0, 1}
        """
        # ── Load image ────────────────────────────────────────────
        image = load_tif(self.image_paths[idx])   # (H, W, 2)
        image = preprocess_sar(image, already_db=self.already_db)

        # ── Load mask ─────────────────────────────────────────────
        mask = load_tif(self.mask_paths[idx])      # (H, W, 1)
        mask = mask[..., 0]                        # (H, W)

        # Invalid pixels (label == -1) are treated as non-flooded (0)
        # for training purposes; they are excluded from metric computation
        # in evaluate.py via a validity mask.
        mask = np.where(mask == 1, 1.0, 0.0).astype(np.float32)

        # ── Apply transforms ──────────────────────────────────────
        augmented = self.transforms(image=image, mask=mask)
        image = augmented["image"]   # (2, H, W) due to ToTensorV2
        mask  = augmented["mask"]    # (H, W)

        # ToTensorV2 keeps mask as (H, W); add channel dim → (1, H, W)
        mask = mask.unsqueeze(0).float()

        return image, mask


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    root_dir: str,
    batch_size: int = 8,
    image_size: int = 512,
    num_workers: int = 4,
    already_db: bool = False
) -> dict:
    """
    Construct and return DataLoaders for train, val, and test splits.

    Parameters
    ----------
    root_dir : str
        Root directory of the Sen1Floods11 dataset.
    batch_size : int
        Number of chips per batch.
    image_size : int
        Spatial resolution of each chip (pixels).
    num_workers : int
        Number of CPU workers for data loading.
    already_db : bool
        Whether the TIF files store SAR values already in dB.

    Returns
    -------
    dict
        Keys: 'train', 'val', 'test' → corresponding DataLoader objects.
    """
    train_ds = Sen1Floods11Dataset(
        root_dir, split="train",
        transforms=get_train_transforms(image_size),
        already_db=already_db
    )
    val_ds = Sen1Floods11Dataset(
        root_dir, split="val",
        transforms=get_val_transforms(image_size),
        already_db=already_db
    )
    test_ds = Sen1Floods11Dataset(
        root_dir, split="test",
        transforms=get_val_transforms(image_size),
        already_db=already_db
    )

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        ),
        "test": DataLoader(
            test_ds, batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
    }
    return loaders
