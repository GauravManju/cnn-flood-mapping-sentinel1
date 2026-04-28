"""
main.py
-------
End-to-end pipeline for the CNN-based flood segmentation project.

Usage
-----
Train the proposed ResNet-34 U-Net model (default):
    python main.py

Train a specific model:
    python main.py --model resnet34_unet
    python main.py --model unet_scratch
    python main.py --model vanilla_cnn

Evaluate a saved checkpoint without training:
    python main.py --eval_only --checkpoint outputs/best_model.pth

Perform threshold sweep on validation set:
    python main.py --threshold_sweep --checkpoint outputs/best_model.pth

Run all three models sequentially (ablation study):
    python main.py --ablation

See all options:
    python main.py --help

Dataset path
------------
Set the path to your Sen1Floods11 root directory via --data_root or
by editing DATA_ROOT below.  The expected structure is:

    <DATA_ROOT>/
      v1.1/
        data/flood_events/HandLabeled/S1Hand/     <- SAR chips
        data/flood_events/HandLabeled/LabelHand/  <- label masks
        splits/flood_handlabeled_split.csv         <- official split

Download the dataset from:
  https://github.com/cloudtostreet/Sen1Floods11
  or Google Cloud Storage: gs://sen1floods11
"""

import os
import argparse
import json
import torch

from src.dataset  import get_dataloaders
from src.model    import build_model
from src.train    import train
from src.evaluate import evaluate, threshold_sweep
from src.utils    import set_seed, get_device, log, load_checkpoint


# ---------------------------------------------------------------------------
# >>>  SET YOUR DATASET PATH HERE  <<<
# ---------------------------------------------------------------------------
DATA_ROOT = "data/sen1floods11"
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Default training configuration (matches report methodology)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # Model
    "model":          "resnet34_unet",  # resnet34_unet | unet_scratch | vanilla_cnn

    # Data
    "data_root":      DATA_ROOT,
    "batch_size":     8,
    "image_size":     512,
    "num_workers":    4,
    "already_db":     False,   # True if TIF files already in dB

    # Training
    "max_epochs":     100,
    "encoder_lr":     1e-5,
    "decoder_lr":     1e-4,
    "weight_decay":   1e-2,
    "bce_weight":     0.5,
    "pos_weight":     4.5,
    "patience":       15,
    "scheduler_t0":   20,
    "threshold":      0.5,

    # Reproducibility
    "seed":           42,

    # I/O
    "output_dir":     "outputs",
    "save_visuals":   True,
    "n_visual_samples": 8
}


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CNN Flood Segmentation — Sen1Floods11"
    )
    p.add_argument("--model",      type=str,   default=None,
                   help="Model name: resnet34_unet | unet_scratch | vanilla_cnn")
    p.add_argument("--data_root",  type=str,   default=None,
                   help="Path to Sen1Floods11 dataset root.")
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--max_epochs", type=int,   default=None)
    p.add_argument("--seed",       type=int,   default=None)
    p.add_argument("--output_dir", type=str,   default=None)
    p.add_argument("--checkpoint", type=str,   default=None,
                   help="Path to .pth checkpoint for evaluation or fine-tuning.")
    p.add_argument("--eval_only",       action="store_true",
                   help="Skip training; evaluate the given checkpoint on the test set.")
    p.add_argument("--threshold_sweep", action="store_true",
                   help="Run threshold sweep on the validation set.")
    p.add_argument("--ablation",        action="store_true",
                   help="Train all three models sequentially.")
    p.add_argument("--no_visuals",      action="store_true",
                   help="Disable saving prediction visualisations.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core pipeline: one model
# ---------------------------------------------------------------------------

def run_pipeline(config: dict, device: torch.device) -> dict:
    """
    Execute the full train → evaluate pipeline for a single model.

    Parameters
    ----------
    config : dict   Configuration dictionary (see DEFAULT_CONFIG).
    device : torch.device

    Returns
    -------
    dict  Final test metrics.
    """
    log(f"{'='*60}")
    log(f"Model: {config['model']}")
    log(f"Output directory: {config['output_dir']}")
    log(f"{'='*60}")

    # ── Set seed ──────────────────────────────────────────────────────────
    set_seed(config["seed"])

    # ── Data ──────────────────────────────────────────────────────────────
    log("Loading dataset ...")
    loaders = get_dataloaders(
        root_dir    = config["data_root"],
        batch_size  = config["batch_size"],
        image_size  = config["image_size"],
        num_workers = config["num_workers"],
        already_db  = config["already_db"]
    )
    log(f"  Train batches : {len(loaders['train'])}")
    log(f"  Val   batches : {len(loaders['val'])}")
    log(f"  Test  batches : {len(loaders['test'])}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(config["model"], device)

    checkpoint_path = os.path.join(config["output_dir"], "best_model.pth")

    # ── (Optional) Load checkpoint before training ─────────────────────
    if config.get("checkpoint"):
        load_checkpoint(model, config["checkpoint"], device)

    # ── Train ─────────────────────────────────────────────────────────────
    if not config.get("eval_only", False):
        log("Starting training ...")
        history = train(
            model        = model,
            train_loader = loaders["train"],
            val_loader   = loaders["val"],
            config       = config,
            device       = device,
            output_dir   = config["output_dir"]
        )
        # Save history as JSON
        history_path = os.path.join(config["output_dir"], "history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        log(f"Training history saved to {history_path}")

    # ── Load best weights for evaluation ─────────────────────────────────
    log("Loading best checkpoint for test evaluation ...")
    load_checkpoint(model, checkpoint_path, device)

    # ── (Optional) Threshold sweep on validation set ───────────────────
    if config.get("threshold_sweep", False):
        log("Running threshold sweep on validation set ...")
        best_t = threshold_sweep(
            model      = model,
            val_loader = loaders["val"],
            device     = device,
            output_dir = config["output_dir"]
        )
        config["threshold"] = best_t
        log(f"Threshold updated to {best_t:.2f} based on validation IoU.")

    # ── Evaluate on test set ──────────────────────────────────────────────
    log("Evaluating on test set ...")
    test_metrics = evaluate(
        model            = model,
        test_loader      = loaders["test"],
        device           = device,
        threshold        = config["threshold"],
        output_dir       = config["output_dir"],
        save_visuals     = config.get("save_visuals", True),
        n_visual_samples = config.get("n_visual_samples", 8)
    )

    # ── Save final metrics ────────────────────────────────────────────────
    metrics_path = os.path.join(config["output_dir"], "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    log(f"Test metrics saved to {metrics_path}")

    return test_metrics


# ---------------------------------------------------------------------------
# Ablation study: run all three models
# ---------------------------------------------------------------------------

def run_ablation(base_config: dict, device: torch.device) -> None:
    """
    Train and evaluate all three models sequentially for the ablation study.
    Each model's outputs are saved in a dedicated sub-directory.
    """
    models = ["vanilla_cnn", "unet_scratch", "resnet34_unet"]
    results = {}

    for model_name in models:
        cfg = base_config.copy()
        cfg["model"]      = model_name
        cfg["output_dir"] = os.path.join(base_config["output_dir"], model_name)

        metrics = run_pipeline(cfg, device)
        results[model_name] = metrics

    # ── Print comparison table ────────────────────────────────────────────
    log("\n" + "="*70)
    log("ABLATION STUDY RESULTS")
    log("="*70)
    header = f"{'Model':<25} {'IoU':>8} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Accuracy':>10}"
    log(header)
    log("-"*70)
    for mname, m in results.items():
        log(
            f"{mname:<25} {m['iou']:>8.4f} {m['dice']:>8.4f} "
            f"{m['precision']:>10.4f} {m['recall']:>8.4f} {m['accuracy']:>10.4f}"
        )
    log("="*70)

    # Save comparison JSON
    comp_path = os.path.join(base_config["output_dir"], "ablation_results.json")
    with open(comp_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Ablation results saved to {comp_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Merge CLI args into config
    config = DEFAULT_CONFIG.copy()
    if args.model:        config["model"]      = args.model
    if args.data_root:    config["data_root"]  = args.data_root
    if args.batch_size:   config["batch_size"] = args.batch_size
    if args.max_epochs:   config["max_epochs"] = args.max_epochs
    if args.seed:         config["seed"]       = args.seed
    if args.output_dir:   config["output_dir"] = args.output_dir
    if args.checkpoint:   config["checkpoint"] = args.checkpoint
    if args.eval_only:    config["eval_only"]  = True
    if args.threshold_sweep: config["threshold_sweep"] = True
    if args.no_visuals:   config["save_visuals"] = False

    os.makedirs(config["output_dir"], exist_ok=True)
    device = get_device()

    # Save the config for reproducibility
    config_path = os.path.join(config["output_dir"], "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    log(f"Config saved to {config_path}")

    if args.ablation:
        run_ablation(config, device)
    else:
        run_pipeline(config, device)


if __name__ == "__main__":
    main()
