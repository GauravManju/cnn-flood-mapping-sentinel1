import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import albumentations as A
from albumentations.pytorch import ToTensorV2

TRAIN_MEAN_VV = -13.2
TRAIN_STD_VV  =   4.8
TRAIN_MEAN_VH = -20.1
TRAIN_STD_VH  =   5.3

def load_tif(path):
    with rasterio.open(path) as src:
        data = src.read()
    return data.transpose(1, 2, 0).astype(np.float32)

def preprocess_sar(chip):
    """
    Data is already in dB and contains NaN nodata pixels.
    Steps:
      1. Take first 2 bands only
      2. Replace NaN with channel mean (sensible fill)
      3. Clamp to valid dB range
      4. Z-score normalise
    """
    if chip.shape[2] > 2:
        chip = chip[..., :2]

    chip = chip.copy()

    # fill NaN per channel with that channel's mean,
    # or with default dB value if entire channel is NaN
    for c in range(chip.shape[2]):
        ch = chip[..., c]
        nan_mask = np.isnan(ch)
        if nan_mask.all():
            ch[nan_mask] = -20.0  # safe default dB value
        elif nan_mask.any():
            ch[nan_mask] = np.nanmean(ch)
        chip[..., c] = ch

    # clamp to sensible dB range
    chip = np.clip(chip, -50.0, 5.0)

    # z-score normalise
    chip[..., 0] = (chip[..., 0] - TRAIN_MEAN_VV) / TRAIN_STD_VV
    chip[..., 1] = (chip[..., 1] - TRAIN_MEAN_VH) / TRAIN_STD_VH

    return chip.astype(np.float32)

def get_train_transforms(image_size=512):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomCrop(height=384, width=384, p=0.5),
        A.Resize(height=image_size, width=image_size),
        A.GaussNoise(p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        ToTensorV2()
    ])

def get_val_transforms(image_size=512):
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        ToTensorV2()
    ])

class Sen1Floods11Dataset(Dataset):
    IMG_DIR  = os.path.join("v1.1","data","flood_events","HandLabeled","S1Hand")
    MASK_DIR = os.path.join("v1.1","data","flood_events","HandLabeled","LabelHand")
    SPLIT_FILES = {
        "train": "flood_train_data.csv",
        "val":   "flood_valid_data.csv",
        "test":  "flood_test_data.csv"
    }

    def __init__(self, root_dir, split="train", transforms=None,
                 image_size=512):
        assert split in ("train","val","test")
        self.root_dir   = root_dir
        self.transforms = transforms
        self.image_size = image_size

        csv_path = os.path.join(root_dir, "v1.1", "splits",
                                self.SPLIT_FILES[split])
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")

        df = pd.read_csv(csv_path, header=None, names=["label_file"])

        self.mask_paths = [
            os.path.join(root_dir, self.MASK_DIR, f.strip())
            for f in df["label_file"].values
        ]
        self.image_paths = [
            os.path.join(root_dir, self.IMG_DIR,
                         f.strip().replace("LabelHand","S1Hand"))
            for f in df["label_file"].values
        ]

        if self.transforms is None:
            self.transforms = A.Compose([
                A.Resize(height=image_size, width=image_size),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_tif(self.image_paths[idx])
        image = preprocess_sar(image)

        mask = load_tif(self.mask_paths[idx])[..., 0]
        mask = np.where(mask == 1, 1.0, 0.0).astype(np.float32)

        aug   = self.transforms(image=image, mask=mask)
        image = aug["image"]
        mask  = aug["mask"].unsqueeze(0).float()

        return image, mask

def get_dataloaders(root_dir, batch_size=8, image_size=512,
                    num_workers=2, already_db=False):
    train_ds = Sen1Floods11Dataset(root_dir, "train",
                   get_train_transforms(image_size))
    val_ds   = Sen1Floods11Dataset(root_dir, "val",
                   get_val_transforms(image_size))
    test_ds  = Sen1Floods11Dataset(root_dir, "test",
                   get_val_transforms(image_size))
    return {
        "train": DataLoader(train_ds, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            pin_memory=True, drop_last=True),
        "val":   DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True),
        "test":  DataLoader(test_ds, batch_size=1,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    }
