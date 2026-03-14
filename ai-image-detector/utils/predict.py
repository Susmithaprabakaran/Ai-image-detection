"""
AI Image Detector - Inference Engine
======================================
Run prediction on single images, batches, or folders.
"""

import os
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageStat
import torchvision.transforms as transforms
from pathlib import Path
from typing import Union, Dict, List, Tuple
import json

from models.detector import build_model, load_model


# ─────────────────────────────────────────────
# Image Preprocessing
# ─────────────────────────────────────────────

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def preprocess_image(image: Union[str, Image.Image]) -> torch.Tensor:
    """Load and preprocess a single image for inference."""
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
    else:
        raise ValueError("image must be a file path or PIL.Image")

    return INFERENCE_TRANSFORM(img).unsqueeze(0)  # (1, C, H, W)


# ─────────────────────────────────────────────
# Frequency Analysis (bonus heuristic)
# ─────────────────────────────────────────────

def analyze_frequency_artifacts(image: Image.Image) -> Dict[str, float]:
    """
    Analyzes subtle statistical artifacts often present in AI-generated images.
    Returns a dict of heuristic scores (0-1, higher = more likely AI).
    """
    img_gray = image.convert("L")
    arr = np.array(img_gray, dtype=np.float32)

    # 1. FFT analysis – AI images often have unusual high-frequency patterns
    fft = np.fft.fft2(arr)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))
    center = np.array(magnitude.shape) // 2
    h, w = magnitude.shape
    r = min(h, w) // 4
    y, x = np.ogrid[:h, :w]
    mask = (x - center[1])**2 + (y - center[0])**2 <= r**2
    high_freq_ratio = magnitude[~mask].mean() / (magnitude[mask].mean() + 1e-8)

    # 2. Local variance uniformity – AI images tend to have overly smooth regions
    img_small = image.resize((64, 64)).convert("L")
    arr_small = np.array(img_small, dtype=np.float32)
    patches = arr_small.reshape(8, 8, 8, 8)
    patch_vars = patches.var(axis=(2, 3))
    variance_uniformity = 1.0 - (patch_vars.std() / (patch_vars.mean() + 1e-8))
    variance_uniformity = float(np.clip(variance_uniformity, 0, 1))

    # 3. Color distribution sharpness
    img_rgb = np.array(image.resize((128, 128)), dtype=np.float32)
    color_std = img_rgb.std(axis=(0, 1)).mean() / 128.0
    color_score = float(np.clip(1.0 - color_std, 0, 1))

    return {
        "high_freq_ratio": float(np.clip(high_freq_ratio / 2.0, 0, 1)),
        "variance_uniformity": variance_uniformity,
        "color_uniformity": color_score,
    }


# ─────────────────────────────────────────────
# Predictor Class
# ─────────────────────────────────────────────

class AIImagePredictor:
    """
    High-level predictor for AI image detection.

    Usage:
        predictor = AIImagePredictor(checkpoint_path="checkpoints/best_model.pth")
        result = predictor.predict("path/to/image.jpg")
        print(result)
    """

    CLASS_NAMES = ["real", "ai_generated"]

    def __init__(
        self,
        checkpoint_path: str = None,
        arch: str = "efficientnet",
        device: str = None,
        use_heuristics: bool = True,
        heuristic_weight: float = 0.2,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_heuristics = use_heuristics
        self.heuristic_weight = heuristic_weight

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model = load_model(checkpoint_path, arch=arch, device=self.device)
        else:
            # Use untrained model (for demo/testing only)
            print("[!] No checkpoint provided – using randomly initialized model.")
            print("    Train the model first with: python train.py")
            self.model = build_model(arch=arch, pretrained=True).to(self.device)
            self.model.eval()

    @torch.no_grad()
    def predict(self, image: Union[str, Image.Image]) -> Dict:
        """
        Predict whether an image is real or AI-generated.

        Returns:
            {
                "prediction": "ai_generated" | "real",
                "confidence": float (0-1),
                "probabilities": {"real": float, "ai_generated": float},
                "heuristics": dict (optional),
                "verdict": str
            }
        """
        # Load PIL image
        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        # CNN prediction
        tensor = preprocess_image(pil_img).to(self.device)
        probs = self.model.predict_proba(tensor).squeeze().cpu().numpy()

        p_real = float(probs[0])
        p_ai   = float(probs[1])

        # Optional heuristic boost
        heuristics = {}
        if self.use_heuristics:
            heuristics = analyze_frequency_artifacts(pil_img)
            heuristic_ai_score = np.mean(list(heuristics.values()))

            # Weighted blend: CNN (80%) + Heuristics (20%)
            w = self.heuristic_weight
            p_ai_final   = (1 - w) * p_ai   + w * heuristic_ai_score
            p_real_final = 1.0 - p_ai_final
        else:
            p_ai_final   = p_ai
            p_real_final = p_real

        prediction = "ai_generated" if p_ai_final > 0.5 else "real"
        confidence  = max(p_ai_final, p_real_final)

        # Human-readable verdict
        if confidence > 0.9:
            certainty = "very high confidence"
        elif confidence > 0.75:
            certainty = "high confidence"
        elif confidence > 0.6:
            certainty = "moderate confidence"
        else:
            certainty = "low confidence (borderline case)"

        verdict = f"This image appears to be {'AI-generated' if prediction == 'ai_generated' else 'real/human-captured'} ({certainty})."

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "probabilities": {
                "real": round(p_real_final, 4),
                "ai_generated": round(p_ai_final, 4),
            },
            "heuristics": heuristics,
            "verdict": verdict,
        }

    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Predict on a list of image paths."""
        results = []
        for path in image_paths:
            try:
                result = self.predict(path)
                result["path"] = path
                results.append(result)
            except Exception as e:
                results.append({"path": path, "error": str(e)})
        return results

    def predict_folder(self, folder: str, extensions: Tuple = (".jpg", ".jpeg", ".png", ".webp", ".bmp")) -> List[Dict]:
        """Run prediction on all images in a folder."""
        image_paths = [
            str(p) for p in Path(folder).rglob("*")
            if p.suffix.lower() in extensions
        ]
        print(f"[Predict] Found {len(image_paths)} images in {folder}")
        return self.predict_batch(image_paths)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AI Image Detection inference")
    parser.add_argument("input",          type=str, help="Image path or folder")
    parser.add_argument("--checkpoint",   type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--arch",         type=str, default="efficientnet")
    parser.add_argument("--no_heuristic", action="store_true")
    parser.add_argument("--output_json",  type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    predictor = AIImagePredictor(
        checkpoint_path=args.checkpoint,
        arch=args.arch,
        use_heuristics=not args.no_heuristic,
    )

    if os.path.isdir(args.input):
        results = predictor.predict_folder(args.input)
    else:
        results = [predictor.predict(args.input)]
        results[0]["path"] = args.input

    # Print results
    print("\n" + "─"*60)
    for r in results:
        if "error" in r:
            print(f"[ERROR] {r['path']}: {r['error']}")
        else:
            icon = "🤖" if r["prediction"] == "ai_generated" else "📷"
            print(f"{icon} {r.get('path', '')}")
            print(f"   Prediction : {r['prediction'].upper()}")
            print(f"   Confidence : {r['confidence']:.1%}")
            print(f"   Verdict    : {r['verdict']}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[✓] Results saved to {args.output_json}")
