"""
AI Image Detector - Dataset Preparation
=========================================
Downloads and organizes recommended datasets for training.

Supported datasets:
  1. CIFAKE  - 60K real (CIFAR-10) + 60K AI-generated images (Kaggle)
  2. Custom  - Organize your own images into the required folder structure

Usage:
  python utils/prepare_dataset.py --dataset cifake --kaggle_user YOUR_USERNAME
  python utils/prepare_dataset.py --dataset custom --real_dir /path/to/real --ai_dir /path/to/ai
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
# CIFAKE via KaggleHub
# ─────────────────────────────────────────────

def download_cifake(output_dir: str = "data") -> None:
    """
    Downloads CIFAKE dataset from Kaggle.
    Requires kaggle API credentials in ~/.kaggle/kaggle.json
    or env vars KAGGLE_USERNAME / KAGGLE_KEY.

    Dataset: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
    """
    try:
        import kaggle
    except ImportError:
        print("[!] kaggle package not installed. Run: pip install kaggle")
        return

    print("[*] Downloading CIFAKE dataset from Kaggle…")
    kaggle.api.dataset_download_files(
        "birdy654/cifake-real-and-ai-generated-synthetic-images",
        path="raw_cifake",
        unzip=True,
    )

    # CIFAKE structure after unzip:
    #   raw_cifake/train/REAL/, raw_cifake/train/FAKE/
    #   raw_cifake/test/REAL/,  raw_cifake/test/FAKE/

    _reorganize_split(
        src_real="raw_cifake/train/REAL",
        src_ai="raw_cifake/train/FAKE",
        dst_real=f"{output_dir}/train/real",
        dst_ai=f"{output_dir}/train/ai_generated",
        val_fraction=0.1,
        dst_val_real=f"{output_dir}/val/real",
        dst_val_ai=f"{output_dir}/val/ai_generated",
    )

    _copy_dir("raw_cifake/test/REAL",  f"{output_dir}/test/real")
    _copy_dir("raw_cifake/test/FAKE",  f"{output_dir}/test/ai_generated")

    print(f"[✓] CIFAKE prepared under '{output_dir}/'")


# ─────────────────────────────────────────────
# Custom Dataset Organizer
# ─────────────────────────────────────────────

def prepare_custom(
    real_dir: str,
    ai_dir: str,
    output_dir: str = "data",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Organize arbitrary real/AI image directories into the train/val/test split.

    Args:
        real_dir:   Directory containing real images
        ai_dir:     Directory containing AI-generated images
        output_dir: Where to write the organized data
        train_frac: Fraction for training set
        val_frac:   Fraction for validation set (rest → test)
        seed:       Random seed for reproducibility
    """
    random.seed(seed)
    EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def get_images(d):
        return [p for p in Path(d).rglob("*") if p.suffix.lower() in EXTS]

    real_imgs = get_images(real_dir)
    ai_imgs   = get_images(ai_dir)

    print(f"[Data] Found {len(real_imgs)} real images, {len(ai_imgs)} AI images.")

    for images, label in [(real_imgs, "real"), (ai_imgs, "ai_generated")]:
        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)

        splits = {
            "train": images[:n_train],
            "val":   images[n_train:n_train + n_val],
            "test":  images[n_train + n_val:],
        }

        for split, files in splits.items():
            dst = Path(output_dir) / split / label
            dst.mkdir(parents=True, exist_ok=True)
            for src in files:
                shutil.copy2(src, dst / src.name)
            print(f"  {split}/{label}: {len(files)} images")

    print(f"\n[✓] Dataset prepared under '{output_dir}/'")
    _print_summary(output_dir)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _reorganize_split(src_real, src_ai, dst_real, dst_ai,
                      val_fraction, dst_val_real, dst_val_ai):
    real_files = list(Path(src_real).glob("*"))
    ai_files   = list(Path(src_ai).glob("*"))
    random.shuffle(real_files); random.shuffle(ai_files)

    n_val_r = int(len(real_files) * val_fraction)
    n_val_a = int(len(ai_files)   * val_fraction)

    _copy_files(real_files[n_val_r:], dst_real)
    _copy_files(ai_files[n_val_a:],   dst_ai)
    _copy_files(real_files[:n_val_r], dst_val_real)
    _copy_files(ai_files[:n_val_a],   dst_val_ai)


def _copy_files(files, dst):
    Path(dst).mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, Path(dst) / Path(f).name)


def _copy_dir(src, dst):
    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        print(f"[!] Source not found: {src}")


def _print_summary(data_dir):
    print("\nDataset Summary:")
    print(f"{'─'*40}")
    for split in ["train", "val", "test"]:
        for label in ["real", "ai_generated"]:
            p = Path(data_dir) / split / label
            count = len(list(p.glob("*"))) if p.exists() else 0
            print(f"  {split:5} / {label:15}: {count:6d} images")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for AI Image Detector")
    parser.add_argument("--dataset",    choices=["cifake", "custom"], required=True)
    parser.add_argument("--output_dir", default="data")
    # For custom only:
    parser.add_argument("--real_dir",   help="Path to real images directory")
    parser.add_argument("--ai_dir",     help="Path to AI-generated images directory")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac",   type=float, default=0.1)
    args = parser.parse_args()

    if args.dataset == "cifake":
        download_cifake(output_dir=args.output_dir)
    elif args.dataset == "custom":
        if not args.real_dir or not args.ai_dir:
            parser.error("--real_dir and --ai_dir are required for custom dataset")
        prepare_custom(
            real_dir=args.real_dir,
            ai_dir=args.ai_dir,
            output_dir=args.output_dir,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
        )
