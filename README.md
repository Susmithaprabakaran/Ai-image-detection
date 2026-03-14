With the explosive rise of tools like Midjourney, DALL·E, and Stable Diffusion, synthetic images flood the internet — often indistinguishable from real photographs. This project tackles that problem with a dual-signal detection system:

🧠 EfficientNet-B3 CNN — fine-tuned deep feature extraction
📊 Frequency-Domain Heuristics — catches statistical artifacts that neural networks alone miss
🌐 Web Interface + REST API — drag-and-drop UI ready for deployment


ai-image-detector/
├── 📄 app.py                      ← Flask web application & REST API
├── 📄 train.py                    ← Training script with validation loop
├── 📄 requirements.txt
│
├── 📂 models/
│   └── detector.py                ← EfficientNet-B3 + Lightweight CNN definitions
│
├── 📂 utils/
│   ├── predict.py                 ← Inference engine (CLI + Python API)
│   ├── prepare_dataset.py         ← Dataset download & train/val/test split
│   └── evaluate.py                ← Metrics, plots, ROC curve
│
├── 📂 templates/
│   └── index.html                 ← Drag-and-drop web UI
│
├── 📂 checkpoints/                ← Saved model weights (auto-created)
└── 📂 data/                       ← Dataset (auto-created)
    ├── train/real/
    ├── train/ai_generated/
    ├── val/
    └── test/
