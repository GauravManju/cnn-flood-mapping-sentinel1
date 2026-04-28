import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from src.utils import compute_iou, compute_dice, AverageMeter, \
                      plot_training_curves, save_checkpoint, log


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # apply sigmoid here for Dice
        preds   = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        return 1.0 - (2.0 * intersection + self.smooth) / \
                     (preds.sum() + targets.sum() + self.smooth)


class BCEDiceLoss(nn.Module):
    """
    BCEWithLogitsLoss (AMP-safe) + Dice loss.
    Takes raw logits as input — sigmoid is applied internally.
    """
    def __init__(self, bce_weight=0.5, pos_weight=4.5):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = 1.0 - bce_weight
        self.dice_loss   = DiceLoss()
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


def build_optimiser(model, encoder_lr=1e-5, decoder_lr=1e-4, weight_decay=1e-2):
    inner = model.model if hasattr(model, "model") else model
    encoder_params, other_params = [], []
    for name, param in inner.named_parameters():
        if name.startswith("encoder"):
            encoder_params.append(param)
        else:
            other_params.append(param)
    if not encoder_params:
        return optim.AdamW(model.parameters(), lr=decoder_lr, weight_decay=weight_decay)
    return optim.AdamW([
        {"params": encoder_params, "lr": encoder_lr},
        {"params": other_params,   "lr": decoder_lr}
    ], weight_decay=weight_decay)


def train_one_epoch(model, loader, optimiser, criterion, device, scaler):
    model.train()
    loss_m = AverageMeter(); iou_m = AverageMeter(); dice_m = AverageMeter()

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)
        optimiser.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(images)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimiser)
        scaler.update()

        # convert logits → probs for metrics only
        with torch.no_grad():
            probs = torch.sigmoid(logits)

        bs = images.size(0)
        loss_m.update(loss.item(), bs)
        iou_m.update(compute_iou(probs, masks), bs)
        dice_m.update(compute_dice(probs, masks), bs)

    return loss_m.avg, iou_m.avg, dice_m.avg


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    loss_m = AverageMeter(); iou_m = AverageMeter(); dice_m = AverageMeter()

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(images)
            loss   = criterion(logits, masks)
        probs = torch.sigmoid(logits)
        bs = images.size(0)
        loss_m.update(loss.item(), bs)
        iou_m.update(compute_iou(probs, masks), bs)
        dice_m.update(compute_dice(probs, masks), bs)

    return loss_m.avg, iou_m.avg, dice_m.avg


def train(model, train_loader, val_loader, config, device, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    criterion = BCEDiceLoss(
        bce_weight=config.get("bce_weight", 0.5),
        pos_weight=config.get("pos_weight", 4.5)
    ).to(device)

    optimiser = build_optimiser(
        model,
        encoder_lr   = config.get("encoder_lr",   1e-5),
        decoder_lr   = config.get("decoder_lr",   1e-4),
        weight_decay = config.get("weight_decay", 1e-2)
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimiser, T_0=config.get("scheduler_t0", 20), eta_min=1e-7)

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    max_epochs = config.get("max_epochs", 100)
    patience   = config.get("patience",    15)
    best_iou   = 0.0
    pat_ctr    = 0
    best_path  = os.path.join(output_dir, "best_model.pth")

    history = {k: [] for k in
               ["train_loss","val_loss","train_iou","val_iou","train_dice","val_dice"]}

    log(f"Training up to {max_epochs} epochs (patience={patience})")
    log(f"Checkpoint -> {best_path}")

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        tr_loss, tr_iou, tr_dice = train_one_epoch(
            model, train_loader, optimiser, criterion, device, scaler)
        va_loss, va_iou, va_dice = validate_one_epoch(
            model, val_loader, criterion, device)
        scheduler.step(epoch - 1)

        log(f"Epoch {epoch:03d}/{max_epochs} | "
            f"train_loss={tr_loss:.4f} train_iou={tr_iou:.4f} | "
            f"val_loss={va_loss:.4f} val_iou={va_iou:.4f} | "
            f"{time.time()-t0:.1f}s")

        for k, v in zip(
            ["train_loss","val_loss","train_iou","val_iou","train_dice","val_dice"],
            [tr_loss, va_loss, tr_iou, va_iou, tr_dice, va_dice]
        ):
            history[k].append(v)

        if va_iou > best_iou:
            best_iou = va_iou
            pat_ctr  = 0
            save_checkpoint(model, optimiser, epoch, va_iou, best_path)
            log(f"  New best val_iou={va_iou:.4f} — saved.")
        else:
            pat_ctr += 1
            if pat_ctr >= patience:
                log(f"Early stopping at epoch {epoch}.")
                break

    plot_training_curves(history, os.path.join(output_dir, "training_curves.png"))
    log(f"Best val IoU: {best_iou:.4f}")
    return history
