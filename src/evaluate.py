"""
evaluate.py
-----------
Comprehensive evaluation of a trained flood segmentation model on the
Sen1Floods11 test set.

Metrics computed
----------------
  - Intersection over Union (IoU / Jaccard index)   ← primary metric
  - Dice coefficient (F1 score)
  - Precision
  - Recall (Sensitivity)
  - Overall pixel accuracy
  - Confusion matrix (TP, FP, FN, TN counts)

Additionally produces:
  - Per-batch aggregated results table (CSV)
  - Qualitative prediction visualisations (saved to outputs/)
  - Printed summary table matching report Table 4.1
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Tuple

from src.utils import log


# ---------------------------------------------------------------------------
# Core metric functions (batch-level, operating on raw tensors)
# ---------------------------------------------------------------------------

def _flatten_preds_targets(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Binarise predictions at `threshold` and flatten both tensors to 1-D.

    Parameters
    ----------
    preds     : (B, 1, H, W) float32 predictions in [0, 1]
    targets   : (B, 1, H, W) float32 binary ground truth in {0, 1}
    threshold : float

    Returns
    -------
    pred_flat   : 1-D BoolTensor
    target_flat : 1-D BoolTensor
    """
    pred_bin = (preds >= threshold).bool().view(-1)
    tgt_bin  = (targets >= 0.5).bool().view(-1)
    return pred_bin, tgt_bin


def batch_confusion(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[int, int, int, int]:
    """
    Compute TP, FP, FN, TN for a batch.

    Returns
    -------
    tp, fp, fn, tn : int counts
    """
    p, t = _flatten_preds_targets(preds, targets, threshold)
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    tn = int((~p & ~t).sum())
    return tp, fp, fn, tn


def metrics_from_confusion(tp, fp, fn, tn) -> Dict[str, float]:
    """
    Derive all segmentation metrics from confusion counts.

    Returns
    -------
    dict with keys: iou, dice, precision, recall, accuracy
    """
    eps = 1e-7
    iou       = tp / (tp + fp + fn + eps)
    dice      = (2 * tp) / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    accuracy  = (tp + tn) / (tp + fp + fn + tn + eps)
    return {
        "iou":       round(iou,       4),
        "dice":      round(dice,      4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "accuracy":  round(accuracy,  4)
    }


# ---------------------------------------------------------------------------
# Full test-set evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    output_dir: str = "outputs",
    save_visuals: bool = True,
    n_visual_samples: int = 8
) -> Dict[str, float]:
    """
    Evaluate the model on the entire test set and return a metrics dictionary.

    Evaluation is performed globally — i.e., confusion counts are accumulated
    across all test batches before metrics are derived — consistent with the
    reporting convention of Bonafilia et al. (2020).

    Parameters
    ----------
    model            : nn.Module   Trained segmentation model.
    test_loader      : DataLoader  Test split loader (batch_size=1 recommended).
    device           : torch.device
    threshold        : float       Binarisation threshold (default 0.5).
    output_dir       : str         Directory to save results.
    save_visuals     : bool        Whether to save qualitative prediction images.
    n_visual_samples : int         Number of samples to visualise.

    Returns
    -------
    metrics : dict
        Keys: 'iou', 'dice', 'precision', 'recall', 'accuracy'
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    total_tp = total_fp = total_fn = total_tn = 0
    visual_count = 0
    batch_records = []   # for per-batch CSV export

    for batch_idx, (images, masks) in enumerate(test_loader):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            preds = model(images)   # (B, 1, H, W)

        tp, fp, fn, tn = batch_confusion(preds, masks, threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        batch_m = metrics_from_confusion(tp, fp, fn, tn)
        batch_records.append({"batch": batch_idx, **batch_m})

        # ── Save qualitative visualisations ───────────────────────
        if save_visuals and visual_count < n_visual_samples:
            _save_prediction_visual(
                images[0].cpu(),
                masks[0].cpu(),
                preds[0].cpu(),
                idx=batch_idx,
                threshold=threshold,
                output_dir=output_dir
            )
            visual_count += 1

    # ── Global metrics ────────────────────────────────────────────────────
    global_metrics = metrics_from_confusion(
        total_tp, total_fp, total_fn, total_tn
    )

    # ── Print results table ───────────────────────────────────────────────
    _print_results_table(global_metrics, total_tp, total_fp, total_fn, total_tn)

    # ── Save confusion matrix visualisation ───────────────────────────────
    _save_confusion_matrix(
        total_tp, total_fp, total_fn, total_tn,
        output_dir=output_dir
    )

    # ── Save per-batch CSV ────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "per_batch_metrics.csv")
    pd.DataFrame(batch_records).to_csv(csv_path, index=False)
    log(f"Per-batch metrics saved to {csv_path}")

    return global_metrics


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _save_prediction_visual(
    image: torch.Tensor,
    mask: torch.Tensor,
    pred: torch.Tensor,
    idx: int,
    threshold: float,
    output_dir: str
) -> None:
    """
    Save a 3-panel figure: VV channel | ground truth | prediction.

    Parameters
    ----------
    image     : (2, H, W) normalised SAR tensor (CPU)
    mask      : (1, H, W) ground truth mask (CPU)
    pred      : (1, H, W) predicted probability map (CPU)
    idx       : int        Sample index for filename.
    threshold : float
    output_dir: str
    """
    vv  = image[0].numpy()    # VV channel (normalised)
    gt  = mask[0].numpy()     # ground truth binary mask
    pr  = pred[0].numpy()     # predicted probability
    pr_bin = (pr >= threshold).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Test Sample {idx:04d}", fontsize=13, fontweight="bold")

    # Panel 1: VV SAR backscatter (inverse normalisation not required for viz)
    axes[0].imshow(vv, cmap="gray", interpolation="nearest")
    axes[0].set_title("Sentinel-1 VV channel\n(normalised)")
    axes[0].axis("off")

    # Panel 2: Ground truth
    axes[1].imshow(gt, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("Ground Truth\n(blue = flooded)")
    axes[1].axis("off")

    # Panel 3: Prediction
    axes[2].imshow(pr_bin, cmap="Blues", vmin=0, vmax=1,
                   interpolation="nearest")
    axes[2].set_title(f"Prediction (threshold={threshold})\n(blue = flooded)")
    axes[2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"prediction_{idx:04d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_confusion_matrix(
    tp: int, fp: int, fn: int, tn: int,
    output_dir: str
) -> None:
    """
    Save a 2×2 confusion matrix heatmap as a PNG.
    """
    cm = np.array([[tn, fp], [fn, tp]])
    labels = [["TN", "FP"], ["FN", "TP"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nNon-Flood", "Predicted\nFlood"])
    ax.set_yticklabels(["Actual\nNon-Flood", "Actual\nFlood"])
    ax.set_title("Pixel-Level Confusion Matrix\n(Test Set)", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Pixel count")

    total = tp + fp + fn + tn
    for i in range(2):
        for j in range(2):
            count   = cm[i, j]
            percent = 100.0 * count / total
            ax.text(j, i,
                    f"{labels[i][j]}\n{count:,}\n({percent:.1f}%)",
                    ha="center", va="center", fontsize=10,
                    color="black" if cm[i, j] < cm.max() * 0.6 else "white")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Confusion matrix saved to {save_path}")


def _print_results_table(
    metrics: Dict[str, float],
    tp: int, fp: int, fn: int, tn: int
) -> None:
    """Print a formatted evaluation summary to stdout."""
    sep = "─" * 52
    log(f"\n{'EVALUATION RESULTS':^52}")
    log(sep)
    log(f"{'Metric':<25} {'Value':>10}")
    log(sep)
    for k, v in metrics.items():
        log(f"{k.capitalize():<25} {v:>10.4f}")
    log(sep)
    log(f"{'True Positives':<25} {tp:>10,}")
    log(f"{'False Positives':<25} {fp:>10,}")
    log(f"{'False Negatives':<25} {fn:>10,}")
    log(f"{'True Negatives':<25} {tn:>10,}")
    log(sep)


# ---------------------------------------------------------------------------
# Threshold sweep (optional — find optimal binarisation threshold)
# ---------------------------------------------------------------------------

@torch.no_grad()
def threshold_sweep(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    thresholds: np.ndarray = None,
    output_dir: str = "outputs"
) -> float:
    """
    Sweep binarisation thresholds on the validation set and return the
    threshold that maximises IoU.  Saves a threshold-vs-IoU plot.

    Parameters
    ----------
    model       : nn.Module
    val_loader  : DataLoader
    device      : torch.device
    thresholds  : np.ndarray  Thresholds to evaluate (default: 0.1 – 0.9).
    output_dir  : str

    Returns
    -------
    best_threshold : float
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # Collect all predictions and targets once
    all_preds  = []
    all_targets = []

    for images, masks in val_loader:
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            preds = model(images).cpu()
        all_preds.append(preds)
        all_targets.append(masks)

    all_preds   = torch.cat(all_preds,   dim=0)   # (N, 1, H, W)
    all_targets = torch.cat(all_targets, dim=0)   # (N, 1, H, W)

    ious = []
    for t in thresholds:
        tp, fp, fn, tn = batch_confusion(all_preds, all_targets, threshold=t)
        m = metrics_from_confusion(tp, fp, fn, tn)
        ious.append(m["iou"])

    best_idx   = int(np.argmax(ious))
    best_t     = float(thresholds[best_idx])
    best_iou   = ious[best_idx]

    log(f"Threshold sweep: best threshold={best_t:.2f}  IoU={best_iou:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, ious, marker="o", color="steelblue", linewidth=2)
    ax.axvline(x=best_t, color="red", linestyle="--",
               label=f"Best threshold = {best_t:.2f}")
    ax.set_xlabel("Binarisation threshold")
    ax.set_ylabel("Validation IoU")
    ax.set_title("Threshold vs. IoU (validation set)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "threshold_sweep.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Threshold sweep plot saved to {plot_path}")

    return best_t
