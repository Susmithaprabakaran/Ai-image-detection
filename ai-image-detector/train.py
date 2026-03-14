"""
AI Image Detector - Training Script
====================================
Train the model on a dataset of real vs AI-generated images.

Dataset structure expected:
    data/
    ├── train/
    │   ├── real/        ← real (human-captured) images
    │   └── ai_generated/ ← AI-generated images
    ├── val/
    │   ├── real/
    │   └── ai_generated/
    └── test/
        ├── real/
        └── ai_generated/

Recommended public datasets:
  - CIFAKE: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
  - ArtiFact: https://github.com/awsaf49/artifact
  - GenImage: https://github.com/GenImage-Dataset/GenImage
"""

import os
import time
import copy
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from models.detector import build_model


# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────

def get_transforms(input_size: int = 224):
    """Returns train/val transforms."""
    train_tf = transforms.Compose([
        transforms.Resize((input_size + 32, input_size + 32)),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    return train_tf, val_tf


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 20 == 0:
            print(f"  Epoch [{epoch}] Step [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f} Acc: {correct/total:.4f}")

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] Using: {device}")

    # ── Data ──
    train_tf, val_tf = get_transforms(args.input_size)

    train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_tf)
    val_dataset   = datasets.ImageFolder(os.path.join(args.data_dir, "val"),   transform=val_tf)
    test_dataset  = datasets.ImageFolder(os.path.join(args.data_dir, "test"),  transform=val_tf)

    class_names = train_dataset.classes
    print(f"[Data] Classes: {class_names}")
    print(f"       Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ── Model ──
    model = build_model(arch=args.arch, pretrained=True).to(device)
    print(f"[Model] Architecture: {args.arch}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"        Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # ── Loss & Optimizer ──
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training Loop ──
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  Starting training for {args.epochs} epochs")
    print(f"{'─'*60}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"\nEpoch [{epoch:03d}/{args.epochs}] ({elapsed:.1f}s) "
              f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_model_wts,
                "val_acc": val_acc,
                "class_names": class_names,
                "arch": args.arch,
            }, os.path.join(args.save_dir, "best_model.pth"))
            print(f"  ★ New best model saved (val_acc={val_acc:.4f})")

    # ── Final Test Evaluation ──
    print(f"\n{'─'*60}")
    print("  Final Evaluation on Test Set")
    print(f"{'─'*60}")
    model.load_state_dict(best_model_wts)
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))

    # Save history
    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[✓] Training complete. Best Val Acc: {best_val_acc:.4f}")
    print(f"    Model saved to: {args.save_dir}/best_model.pth")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI Image Detector")
    parser.add_argument("--data_dir",    type=str, default="data",       help="Root data directory")
    parser.add_argument("--save_dir",    type=str, default="checkpoints", help="Where to save models")
    parser.add_argument("--arch",        type=str, default="efficientnet", choices=["efficientnet", "lightweight"])
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--input_size",  type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    train(args)
