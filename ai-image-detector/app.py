"""
AI Image Detector - Flask Web Application
==========================================
Run: python app.py
Then visit: http://localhost:5000
"""

import os
import io
import base64
import uuid
import json
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import torch

from utils.predict import AIImagePredictor

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload
app.config["UPLOAD_FOLDER"] = "static/uploads"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

CHECKPOINT_PATH = os.getenv("MODEL_CHECKPOINT", "checkpoints/best_model.pth")
MODEL_ARCH      = os.getenv("MODEL_ARCH", "efficientnet")

# Load model at startup
predictor = AIImagePredictor(
    checkpoint_path=CHECKPOINT_PATH,
    arch=MODEL_ARCH,
    use_heuristics=True,
)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts:
      - multipart/form-data with key 'image'
      - JSON with key 'image_base64' (base64-encoded image)

    Returns JSON with detection result.
    """
    pil_image = None

    # ── Handle file upload ──
    if "image" in request.files:
        file = request.files["image"]
        if not file or file.filename == "":
            return jsonify({"error": "No file provided"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Unsupported file type"}), 400

        try:
            pil_image = Image.open(file.stream).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Cannot open image: {str(e)}"}), 400

    # ── Handle base64 ──
    elif request.is_json and "image_base64" in request.json:
        try:
            data = request.json["image_base64"]
            if "," in data:
                data = data.split(",")[1]
            img_bytes = base64.b64decode(data)
            pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Invalid base64 image: {str(e)}"}), 400

    else:
        return jsonify({"error": "No image provided. Use 'image' (file) or 'image_base64' (JSON)."}), 400

    # ── Run prediction ──
    try:
        result = predictor.predict(pil_image)
        result["model_info"] = {
            "architecture": MODEL_ARCH,
            "device": str(predictor.device),
            "heuristics_enabled": predictor.use_heuristics,
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model": MODEL_ARCH,
        "device": str(predictor.device),
        "checkpoint": CHECKPOINT_PATH,
        "torch_version": torch.__version__,
    })


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"\n🚀 AI Image Detector running at http://localhost:{port}")
    print(f"   Model: {MODEL_ARCH} | Device: {predictor.device}")
    app.run(host="0.0.0.0", port=port, debug=debug)
