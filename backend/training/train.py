"""
Training script: EfficientNet-B4 fine-tuned for 3-class fake image detection.
  Classes: REAL (0) | AI_GENERATED (1) | EDITED (2)

Usage:
    python training/train.py --data ../data --epochs 30 --batch-size 32

Two-phase training:
    Phase 1 (epochs 1--freeze-epochs): Freeze backbone, train classifier head only
    Phase 2 (remaining epochs):        Unfreeze all layers, fine-tune with lower LR
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.cnn_model import FakeImageDetector, CLASSES, NUM_CLASSES
from training.dataset import FakeImageDataset

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ─── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Multi-class precision/recall/F1 (macro-averaged) + per-class breakdown."""
    preds = logits.argmax(dim=1)
    acc = (preds == labels).float().mean().item()

    per_class = {}
    f1_scores = []
    for c in range(NUM_CLASSES):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        per_class[CLASSES[c]] = {"precision": prec, "recall": rec, "f1": f1}
        f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores))
    macro_prec = float(np.mean([v["precision"] for v in per_class.values()]))
    macro_rec  = float(np.mean([v["recall"]    for v in per_class.values()]))

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "per_class": per_class,
    }


# ─── train / eval loops ────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, all_logits, all_labels = 0.0, [], []

    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.cpu())

        if (step + 1) % 50 == 0:
            print(f"  step {step+1}/{len(loader)}  loss={loss.item():.4f}")

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(loader)
    return metrics


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_logits, all_labels = 0.0, [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)
        total_loss += loss.item()
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(loader)
    return metrics


def print_metrics(tag: str, m: dict, elapsed: float = 0):
    print(f"{tag} | loss={m['loss']:.4f}  acc={m['acc']:.4f}  "
          f"macro_f1={m['macro_f1']:.4f}  prec={m['macro_precision']:.4f}  rec={m['macro_recall']:.4f}"
          + (f"  [{elapsed:.0f}s]" if elapsed else ""))
    for cls, scores in m["per_class"].items():
        print(f"       {cls:14s}  f1={scores['f1']:.3f}  prec={scores['precision']:.3f}  rec={scores['recall']:.3f}")


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",          default="../data")
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--batch-size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--lr-fine",       type=float, default=2e-5)
    parser.add_argument("--workers",       type=int,   default=4)
    parser.add_argument("--freeze-epochs", type=int,   default=5)
    parser.add_argument("--resume",        default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Classes: {CLASSES}")
    if device.type == "cpu":
        print("WARNING: Training on CPU will be very slow. Use a GPU.")

    train_ds = FakeImageDataset(args.data, split="train")
    val_ds   = FakeImageDataset(args.data, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_ds.make_sampler(),
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda",
    )

    model = FakeImageDetector(pretrained=True).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        print(f"Resumed from {args.resume}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()
    best_f1 = 0.0
    scheduler = None

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.epochs}")

        if epoch <= args.freeze_epochs:
            if epoch == 1:
                print("Phase 1: training head only (backbone frozen)")
                model.freeze_backbone()
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr, weight_decay=1e-4,
            )
        elif epoch == args.freeze_epochs + 1:
            print("Phase 2: unfreezing backbone for full fine-tuning")
            model.unfreeze_all()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr_fine, weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.freeze_epochs, eta_min=1e-7,
            )

        t0 = time.time()
        train_m = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_m   = eval_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        if scheduler and epoch > args.freeze_epochs:
            scheduler.step()

        print()
        print_metrics("Train", train_m)
        print_metrics("Val  ", val_m, elapsed)

        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            ckpt_path = CHECKPOINT_DIR / "efficientnet_b4_fake.pth"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "val_macro_f1": best_f1,
                "val_acc": val_m["acc"],
                "classes": CLASSES,
            }, ckpt_path)
            print(f"  ✓ Best checkpoint saved (macro_f1={best_f1:.4f}) → {ckpt_path}")

    print(f"\nTraining complete. Best val macro-F1: {best_f1:.4f}")
    print(f"Checkpoint: {CHECKPOINT_DIR / 'efficientnet_b4_fake.pth'}")


if __name__ == "__main__":
    main()
