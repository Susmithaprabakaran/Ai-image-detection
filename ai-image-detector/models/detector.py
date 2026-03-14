"""
AI Image Detector - CNN Model
Detects whether an image is AI-generated or real (human-captured).
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


# ─────────────────────────────────────────────
# Model Definition
# ─────────────────────────────────────────────

class AIImageDetector(nn.Module):
    """
    Binary classifier using a fine-tuned EfficientNet-B3 backbone.
    Outputs probability of being AI-generated.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.4):
        super(AIImageDetector, self).__init__()

        # Load EfficientNet-B3 backbone
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b3(weights=weights)

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)   # [real, ai_generated]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities [p_real, p_ai]."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


# ─────────────────────────────────────────────
# Alternative: Lightweight Custom CNN
# (use when EfficientNet is too heavy)
# ─────────────────────────────────────────────

class LightweightDetector(nn.Module):
    """
    Fast, lightweight CNN for AI image detection.
    Good for edge deployment or quick experiments.
    """

    def __init__(self, dropout: float = 0.3):
        super(LightweightDetector, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),

            # Block 4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


# ─────────────────────────────────────────────
# Model Factory
# ─────────────────────────────────────────────

def build_model(arch: str = "efficientnet", pretrained: bool = True) -> nn.Module:
    """
    Factory function to create the detection model.

    Args:
        arch: 'efficientnet' or 'lightweight'
        pretrained: whether to use ImageNet pretrained weights for backbone

    Returns:
        nn.Module
    """
    if arch == "efficientnet":
        return AIImageDetector(pretrained=pretrained)
    elif arch == "lightweight":
        return LightweightDetector()
    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose 'efficientnet' or 'lightweight'.")


def load_model(checkpoint_path: str, arch: str = "efficientnet", device: str = "cpu") -> nn.Module:
    """
    Load a trained model from a checkpoint file.

    Args:
        checkpoint_path: path to .pth checkpoint
        arch: model architecture used during training
        device: 'cuda' or 'cpu'

    Returns:
        Loaded nn.Module in eval mode
    """
    model = build_model(arch=arch, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support both raw state_dict and wrapped checkpoints
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print(f"[✓] Model loaded from {checkpoint_path} on {device}")
    return model
