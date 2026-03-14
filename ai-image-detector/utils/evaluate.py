"""
AI Image Detector - Evaluation & Visualization
================================================
Evaluates a trained model and generates a detailed report with plots.

Usage:
  python utils/evaluate.py --checkpoint checkpoints/best_model.pth --data_dir data
"""

import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

# Optional: matplotlib for plots
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[!] matplotlib/seaborn not installed. Skipping plots.")

from models.detector import load_model


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for images, labels in loader:
        images = images.to(device)
        probs  = model.predict_proba(images).cpu().numpy()
        preds  = probs.argmax(axis=1)

        all_probs.extend(probs[:, 1])   # P(ai_generated)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("Confusion Matrix", fontsize=14)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Confusion matrix saved to {save_path}")


def plot_roc_curve(labels, probs, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#00e5ff", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] ROC curve saved to {save_path}")


def plot_training_history(history_path, save_path):
    with open(history_path) as f:
        h = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(h["train_loss"], label="Train Loss", color="#00e5ff")
    ax1.plot(h["val_loss"],   label="Val Loss",   color="#ff4d6d")
    ax1.set_title("Loss Curve"); ax1.legend()
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")

    ax2.plot(h["train_acc"], label="Train Acc", color="#00e5ff")
    ax2.plot(h["val_acc"],   label="Val Acc",   color="#ff4d6d")
    ax2.set_title("Accuracy Curve"); ax2.legend()
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Training history saved to {save_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint, arch=args.arch, device=str(device))

    # Data
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset      = datasets.ImageFolder(os.path.join(args.data_dir, "test"), transform=tf)
    class_names  = dataset.classes
    loader       = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"[Eval] Classes: {class_names} | Test samples: {len(dataset)}")

    labels, preds, probs = run_eval(model, loader, device)

    # ── Metrics ──
    print("\n" + "─"*60)
    print("Classification Report:")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    auc = roc_auc_score(labels, probs)
    print(f"ROC-AUC Score: {auc:.4f}")
    print("─"*60)

    # Save metrics JSON
    metrics = {
        "accuracy": float((labels == preds).mean()),
        "roc_auc": float(auc),
        "report": classification_report(labels, preds, target_names=class_names, output_dict=True),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Plots ──
    if HAS_PLOT:
        plot_confusion_matrix(labels, preds, class_names,
                              os.path.join(args.output_dir, "confusion_matrix.png"))
        plot_roc_curve(labels, probs,
                       os.path.join(args.output_dir, "roc_curve.png"))

        history_path = os.path.join(os.path.dirname(args.checkpoint), "history.json")
        if os.path.exists(history_path):
            plot_training_history(history_path,
                                  os.path.join(args.output_dir, "training_history.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AI Image Detector")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--data_dir",    type=str, default="data")
    parser.add_argument("--arch",        type=str, default="efficientnet")
    parser.add_argument("--output_dir",  type=str, default="eval_results")
    args = parser.parse_args()
    main(args)
