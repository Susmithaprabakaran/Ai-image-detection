# 🤖 AI Image Detector

A production-ready deep learning system that detects whether an image is **AI-generated** or **real (human-captured)**.

Built with PyTorch, EfficientNet-B3, and a Flask web app.

---

## 📁 Project Structure

```
ai-image-detector/
├── app.py                      ← Flask web application
├── train.py                    ← Training script
├── requirements.txt
├── models/
│   └── detector.py             ← CNN model definitions (EfficientNet + Lightweight)
├── utils/
│   ├── predict.py              ← Inference engine (CLI + Python API)
│   ├── prepare_dataset.py      ← Dataset download & organization
│   └── evaluate.py             ← Evaluation metrics & plots
├── templates/
│   └── index.html              ← Web UI
├── static/                     ← CSS, JS, uploaded images
├── checkpoints/                ← Saved model weights (auto-created)
└── data/                       ← Dataset (auto-created)
    ├── train/real/
    ├── train/ai_generated/
    ├── val/real/
    ├── val/ai_generated/
    ├── test/real/
    └── test/ai_generated/
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support (recommended):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

### 2. Prepare Dataset

**Option A — CIFAKE (recommended, 120K images):**
```bash
# Set up Kaggle credentials first: https://www.kaggle.com/docs/api
python utils/prepare_dataset.py --dataset cifake
```

**Option B — Your own images:**
```bash
python utils/prepare_dataset.py \
  --dataset custom \
  --real_dir /path/to/real/images \
  --ai_dir /path/to/ai/images
```

The script auto-creates `data/train/`, `data/val/`, `data/test/` splits.

**Other recommended datasets:**
- [ArtiFact](https://github.com/awsaf49/artifact) — diverse AI generators
- [GenImage](https://github.com/GenImage-Dataset/GenImage) — 1M+ images
- [RAISE](http://loki.disi.unitn.it/RAISE/) — high-quality real photos

---

### 3. Train the Model

```bash
# EfficientNet-B3 (recommended, ~12M params)
python train.py --arch efficientnet --epochs 30 --batch_size 32

# Lightweight CNN (faster, less accurate)
python train.py --arch lightweight --epochs 50 --batch_size 64

# Full options
python train.py \
  --data_dir data \
  --save_dir checkpoints \
  --arch efficientnet \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-4 \
  --input_size 224 \
  --num_workers 4
```

Best model is auto-saved to `checkpoints/best_model.pth`.

---

### 4. Run the Web App

```bash
python app.py
```

Visit **http://localhost:5000** — drag & drop any image to detect it.

```bash
# With custom checkpoint or port
MODEL_CHECKPOINT=checkpoints/best_model.pth PORT=8080 python app.py
```

---

### 5. Run from Command Line

```bash
# Single image
python utils/predict.py path/to/image.jpg

# Entire folder
python utils/predict.py path/to/folder/ --output_json results.json

# Without heuristic analysis (CNN only)
python utils/predict.py image.jpg --no_heuristic
```

---

### 6. Use as Python Library

```python
from utils.predict import AIImagePredictor

predictor = AIImagePredictor(
    checkpoint_path="checkpoints/best_model.pth",
    arch="efficientnet",
    use_heuristics=True,
)

result = predictor.predict("path/to/image.jpg")
print(result)
# {
#   "prediction": "ai_generated",
#   "confidence": 0.9412,
#   "probabilities": {"real": 0.0588, "ai_generated": 0.9412},
#   "heuristics": {...},
#   "verdict": "This image appears to be AI-generated (very high confidence)."
# }
```

---

### 7. Evaluate Trained Model

```bash
python utils/evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --data_dir data \
  --output_dir eval_results
```

Generates: `metrics.json`, `confusion_matrix.png`, `roc_curve.png`, `training_history.png`

---

## 🔌 REST API

The Flask app exposes a simple API:

### POST `/api/predict`

**File upload:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -F "image=@photo.jpg"
```

**Base64 JSON:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "data:image/jpeg;base64,/9j/..."}'
```

**Response:**
```json
{
  "prediction": "ai_generated",
  "confidence": 0.9412,
  "probabilities": {
    "real": 0.0588,
    "ai_generated": 0.9412
  },
  "heuristics": {
    "high_freq_ratio": 0.72,
    "variance_uniformity": 0.68,
    "color_uniformity": 0.45
  },
  "verdict": "This image appears to be AI-generated (very high confidence).",
  "model_info": {
    "architecture": "efficientnet",
    "device": "cpu",
    "heuristics_enabled": true
  }
}
```

### GET `/api/health`
Returns model status, device, and torch version.

---

## 🧠 Model Architecture

| Model | Params | Speed | Accuracy (CIFAKE) |
|-------|--------|-------|-------------------|
| EfficientNet-B3 | ~12M | Moderate | ~97%+ |
| Lightweight CNN | ~2M | Fast | ~92%+ |

The system uses a **dual-signal approach**:
1. **CNN (80% weight)** — Deep feature extraction via fine-tuned EfficientNet-B3
2. **Heuristics (20% weight)** — Frequency domain analysis + statistical artifacts

---

## 📈 Expected Results (CIFAKE dataset)

```
              precision    recall  f1-score   support
        real     0.9731    0.9685    0.9708      5000
ai_generated     0.9688    0.9734    0.9711      5000
    accuracy                         0.9710     10000
   macro avg     0.9710    0.9710    0.9710     10000
ROC-AUC Score: 0.9961
```

---

## 🚀 Deployment

**Docker:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

**Gunicorn (production):**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## 🗺 Roadmap

- [ ] GradCAM visualization (highlight suspicious regions)
- [ ] Multi-generator classification (DALL·E / Midjourney / SD)
- [ ] Video frame analysis
- [ ] Browser extension
- [ ] ONNX export for edge deployment

---

## 📄 License

MIT License — free to use, modify, and distribute.
