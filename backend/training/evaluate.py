"""
Evaluate a trained 3-class checkpoint on the val set.

Usage:
    python training/evaluate.py --checkpoint checkpoints/efficientnet_b4_fake.pth --data ../data
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.cnn_model import FakeImageDetector, CLASSES, NUM_CLASSES
from training.dataset import FakeImageDataset


def evaluate(checkpoint_path: str, data_dir: str, batch_size: int = 32, workers: int = 4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(checkpoint_path, map_location=device)
    model = FakeImageDetector(pretrained=False).to(device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    print(f"Loaded: {checkpoint_path}")
    if "epoch" in state:
        print(f"Trained {state['epoch']} epochs  |  val_macro_f1={state.get('val_macro_f1', '?')}")

    val_ds = FakeImageDataset(data_dir, split="val")
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)

    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            with autocast():
                logits = model(imgs)
            import torch.nn.functional as F
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    acc = (all_preds == all_labels).mean()

    # Per-class metrics + confusion matrix
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        conf_matrix[int(true)][int(pred)] += 1

    per_class_metrics = {}
    f1_scores = []
    for c in range(NUM_CLASSES):
        tp = conf_matrix[c, c]
        fp = conf_matrix[:, c].sum() - tp
        fn = conf_matrix[c, :].sum() - tp
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        per_class_metrics[CLASSES[c]] = {"precision": float(prec), "recall": float(rec), "f1": float(f1)}
        f1_scores.append(f1)

    macro_f1   = float(np.mean(f1_scores))
    macro_prec = float(np.mean([v["precision"] for v in per_class_metrics.values()]))
    macro_rec  = float(np.mean([v["recall"]    for v in per_class_metrics.values()]))

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS — 3-Class Fake Image Detection")
    print("=" * 60)
    print(f"Accuracy      : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"Macro Precision: {macro_prec:.4f}")
    print(f"Macro Recall  : {macro_rec:.4f}")

    print("\nPer-Class Metrics:")
    header = f"{'Class':>15}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>8}"
    print(header)
    print("-" * len(header))
    for c in range(NUM_CLASSES):
        support = conf_matrix[c, :].sum()
        m = per_class_metrics[CLASSES[c]]
        print(f"{CLASSES[c]:>15}  {m['precision']:>10.4f}  {m['recall']:>8.4f}  {m['f1']:>8.4f}  {support:>8}")

    print("\nConfusion Matrix (rows=True, cols=Predicted):")
    header_cols = "".join(f"{c:>15}" for c in CLASSES)
    print(f"{'':>15}" + header_cols)
    for i, cls in enumerate(CLASSES):
        row = "".join(f"{conf_matrix[i, j]:>15}" for j in range(NUM_CLASSES))
        print(f"{cls:>15}" + row)
    print("=" * 60)

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "per_class": per_class_metrics,
        "confusion_matrix": conf_matrix.tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/efficientnet_b4_fake.pth")
    parser.add_argument("--data",       default="../data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers",    type=int, default=4)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data, args.batch_size, args.workers)
