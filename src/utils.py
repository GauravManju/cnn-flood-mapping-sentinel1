"""
utils.py
--------
Shared utility functions used across the project:

  - compute_iou       : batch-level IoU for training loops
  - compute_dice      : batch-level Dice for training loops
  - AverageMeter      : running mean tracker for epoch metrics
  - set_seed          : reproducibility seed setter
  - log               : timestamped console logger
  - save_checkpoint   : save model + optimiser state
  - load_checkpoint   : restore model from checkpoint
  - plot_training_curves : save training/validation curve figure
  - count_parameters  : count trainable parameters
"""

import os
import random
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(message: str) -> None:
    """
    Print a timestamped log message to stdout.

    Parameters
    ----------
    message : str
        Message to print.
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set all relevant random seeds for reproducibility.

    Covers: Python random, NumPy, PyTorch CPU and GPU, CUDA deterministic
    algorithms.  Note: full determinism may reduce performance on some
    GPU operations.

    Parameters
    ----------
    seed : int
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    log(f"Global random seed set to {seed}")


# ---------------------------------------------------------------------------
# Metrics (batch-level, used in training loops)
# ---------------------------------------------------------------------------

def compute_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7
) -> float:
    """
    Compute Intersection over Union (Jaccard index) for a batch.

    Counts are accumulated across the entire batch before computing the ratio,
    giving a micro-averaged IoU consistent with the global metric used in
    Bonafilia et al. (2020).

    Parameters
    ----------
    preds     : (B, 1, H, W) float32 predictions in [0, 1]
    targets   : (B, 1, H, W) float32 binary ground truth in {0, 1}
    threshold : float   Binarisation threshold
    eps       : float   Numerical stability

    Returns
    -------
    float
    """
    preds_bin = (preds >= threshold).float()
    intersection = (preds_bin * targets).sum()
    union = preds_bin.sum() + targets.sum() - intersection
    return float(intersection / (union + eps))


def compute_dice(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7
) -> float:
    """
    Compute the Sørensen-Dice coefficient (F1 score) for a batch.

    Parameters
    ----------
    preds     : (B, 1, H, W) float32 predictions in [0, 1]
    targets   : (B, 1, H, W) float32 binary ground truth in {0, 1}
    threshold : float
    eps       : float

    Returns
    -------
    float
    """
    preds_bin = (preds >= threshold).float()
    intersection = (preds_bin * targets).sum()
    return float((2.0 * intersection) / (preds_bin.sum() + targets.sum() + eps))


# ---------------------------------------------------------------------------
# Running average tracker
# ---------------------------------------------------------------------------

class AverageMeter:
    """
    Tracks a running mean over a sequence of values.
    Useful for accumulating per-batch metrics over an epoch.

    Attributes
    ----------
    val : float   Most recent value
    avg : float   Running average
    sum : float   Cumulative sum
    count : int   Number of updates
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update the meter.

        Parameters
        ----------
        val : float  Value to record (e.g. loss for a batch).
        n   : int    Batch size (used to weight the mean correctly).
        """
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / (self.count + 1e-9)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimiser,
    epoch: int,
    val_iou: float,
    path: str
) -> None:
    """
    Save model weights, optimiser state, epoch, and best IoU to a file.

    Parameters
    ----------
    model     : nn.Module
    optimiser : torch.optim.Optimizer
    epoch     : int
    val_iou   : float
    path      : str   Full path of the .pth output file.
    """
    state = {
        "epoch":     epoch,
        "val_iou":   val_iou,
        "model":     model.state_dict(),
        "optimiser": optimiser.state_dict()
    }
    torch.save(state, path)


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device,
    optimiser=None
) -> int:
    """
    Load a saved checkpoint into a model (and optionally an optimiser).

    Parameters
    ----------
    model     : nn.Module
    path      : str           Path to the .pth checkpoint file.
    device    : torch.device
    optimiser : optional      If provided, also restores optimiser state.

    Returns
    -------
    epoch : int  The epoch at which the checkpoint was saved.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    if optimiser is not None and "optimiser" in checkpoint:
        optimiser.load_state_dict(checkpoint["optimiser"])

    epoch   = checkpoint.get("epoch",   0)
    val_iou = checkpoint.get("val_iou", 0.0)
    log(f"Loaded checkpoint from '{path}'  (epoch={epoch}, val_iou={val_iou:.4f})")
    return epoch


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot and optionally save training and validation loss, IoU, and Dice
    curves across epochs.

    Parameters
    ----------
    history   : dict   Keys: 'train_loss', 'val_loss', 'train_iou',
                       'val_iou', 'train_dice', 'val_dice'.
    save_path : str or None   If given, saves the figure to this path.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    # ── Loss ──────────────────────────────────────────────────────
    axes[0].plot(epochs, history["train_loss"], label="Train", color="steelblue")
    axes[0].plot(epochs, history["val_loss"],   label="Val",   color="tomato")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (BCE + Dice)")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    # ── IoU ───────────────────────────────────────────────────────
    axes[1].plot(epochs, history["train_iou"], label="Train", color="steelblue")
    axes[1].plot(epochs, history["val_iou"],   label="Val",   color="tomato")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].set_title("Intersection over Union (IoU)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    # ── Dice ──────────────────────────────────────────────────────
    axes[2].plot(epochs, history["train_dice"], label="Train", color="steelblue")
    axes[2].plot(epochs, history["val_dice"],   label="Val",   color="tomato")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Dice / F1")
    axes[2].set_title("Dice Coefficient")
    axes[2].legend()
    axes[2].grid(True, alpha=0.4)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)


def plot_sample_predictions(
    images: torch.Tensor,
    masks: torch.Tensor,
    preds: torch.Tensor,
    n: int = 4,
    threshold: float = 0.5,
    save_path: Optional[str] = None
) -> None:
    """
    Display a grid of sample predictions alongside input VV channel
    and ground truth masks.

    Parameters
    ----------
    images    : (B, 2, H, W)  SAR chip batch
    masks     : (B, 1, H, W)  Ground truth masks
    preds     : (B, 1, H, W)  Predicted probability maps
    n         : int            Number of samples to display (≤ B)
    threshold : float
    save_path : str or None
    """
    n = min(n, images.size(0))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

    if n == 1:
        axes = [axes]

    for i in range(n):
        vv  = images[i, 0].cpu().numpy()
        gt  = masks[i, 0].cpu().numpy()
        pr  = (preds[i, 0].cpu().numpy() >= threshold).astype(np.float32)

        axes[i][0].imshow(vv, cmap="gray")
        axes[i][0].set_title("VV SAR input" if i == 0 else "")
        axes[i][0].axis("off")

        axes[i][1].imshow(gt, cmap="Blues", vmin=0, vmax=1)
        axes[i][1].set_title("Ground truth" if i == 0 else "")
        axes[i][1].axis("off")

        axes[i][2].imshow(pr, cmap="Blues", vmin=0, vmax=1)
        axes[i][2].set_title(f"Prediction (t={threshold})" if i == 0 else "")
        axes[i][2].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> str:
    """
    Return a formatted string summarising trainable and total parameter counts.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return f"Trainable: {trainable:,}  |  Total: {total:,}"


def get_device() -> torch.device:
    """
    Return CUDA if available, else CPU.  Logs the selected device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    if device.type == "cuda":
        log(f"  GPU: {torch.cuda.get_device_name(0)}")
        log(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device
