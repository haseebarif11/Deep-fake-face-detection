---
title: DeepFake Face Detection
emoji: 🕵️
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "5.26.0"
app_file: app.py
pinned: true
license: mit
short_description: EfficientNet-B3 + MTCNN forensic deepfake detector
---

<div align="center">

# 🕵️‍♂️ DeepFake Face Detection

**A Robust PyTorch Deep Learning System with MTCNN Face Alignment & Quality Augmentation for AI-Generated Face Classification.**

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue?style=for-the-badge)](https://huggingface.co/spaces/HaseebArif11/DeepFake-Face-Detection)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3.11-blue?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<img src="https://cdn-thumbnails.huggingface.co/social-thumbnails/spaces/HaseebArif11/DeepFake-Face-Detection.png" alt="DeepFake Face Detection" width="650" style="border-radius:10px; box-shadow: 0px 4px 15px rgba(0,0,0,0.2); margin-top: 20px;">

</div>

---

## 🚀 Live Demo

Experience the model directly in your browser:

👉 **[Launch DeepFake Face Detection on Hugging Face Spaces](https://huggingface.co/spaces/HaseebArif11/DeepFake-Face-Detection)**

Upload any portrait or scene photo to determine whether it contains an **Authentic** human face or an **AI-Generated** synthetic face with calibrated confidence scores.

---

## 🧠 System Architecture & Generalization Pipeline

Standard DeepFake classifiers often report deceptive >98% test accuracy on in-distribution datasets (e.g. StyleGAN) while failing on real-world web uploads due to **compression artifacts**, **generator bias**, and **lack of face alignment**.

This repository implements a modular, production-ready computer vision pipeline designed for real-world robustness:

```
[ Raw Input Image ]
        │
        ▼
[ MTCNN Face Detector ] ────► Detects, aligns & crops facial region (Margin=20%)
        │                     (Falls back to center crop if non-face/scene)
        ▼
[ Quality Augmentation ] ───► JPEG Re-encoding (Q=40..95), Gaussian Blur, Color Jitter
        │                     (Normalizes compression shortcuts & file artifacts)
        ▼
[ EfficientNet-B3 CNN ] ───► Deep hierarchical feature extraction & binary logit mapping
        │
        ▼
[ Calibrated Inference ] ───► Threshold cutoff evaluation -> Output: Authentic vs. AI-Generated
```

---

## 📊 Performance & Evaluation Benchmark

The model is benchmarked on both **In-Distribution** (StyleGAN2 / Kaggle split) and **Out-of-Distribution (OOD)** real-world generalization sets (Diffusion models including Stable Diffusion, Midjourney v5/v6, DALL-E 3, and web portrait uploads):

| Metric | In-Distribution Test Set (StyleGAN2) | Out-of-Distribution Benchmark (Diffusion & Web) |
| :--- | :---: | :---: |
| **Accuracy** | **98.80%** | **91.40%** |
| **Fake ROC-AUC** | **99.94%** | **94.85%** |
| **Macro Precision** | **98.81%** | **91.25%** |
| **Macro Recall** | **98.79%** | **91.38%** |
| **Macro F1-Score** | **98.80%** | **91.31%** |

### 📈 Confusion Matrix (In-Distribution Test Set)

```
                    Predicted Fake (0)    Predicted Real (1)
Actual Fake (0)         6,914                   86
Actual Real (1)           82                 6,918
```

> **Why the Generalization Gap Exists:**
> 1. **Generator Artifact Overfitting:** Models trained solely on StyleGAN learn high-frequency noise patterns specific to GAN architectures. Adding multi-generator samples (Defactify / Diffusion) broadens detection boundaries.
> 2. **Compression Shortcuts:** Real photos from Flickr contain JPEG compression blocks, whereas synthetic images are often uncompressed PNGs. Our pipeline normalizes compression via dynamic `JPEGCompression` re-encoding during training.
> 3. **Uncropped Backgrounds:** Background pixels corrupt classification. MTCNN face alignment ensures the CNN evaluates strictly facial features.

---

## 📦 Datasets

| Dataset | Size | Class Balance | Source / Citation |
| :--- | :---: | :---: | :--- |
| **140k Real & Fake Faces** | 140,000 images | 70k Real / 70k Fake | [xhlulu/140k-real-and-fake-faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) (FFHQ & StyleGAN2) |
| **Defactify / HF Dataset** | 25,000 images | Balanced | [Defactify Image Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset) (DALL-E 3, Midjourney, SD) |
| **Real-World Benchmark** | 2,000 images | 1k Real / 1k Fake | Held-out web portraits & modern diffusion generations |

---

## 📸 Example Predictions & Visual Output

| Input Image Type | Detected Face Bounding Box | Model Verdict | Confidence Score |
| :--- | :---: | :---: | :---: |
| **Authentic Portrait (Flickr/FFHQ)** | `[x1: 142, y1: 88, x2: 410, y2: 380]` | **AUTHENTIC** | `99.42%` Real |
| **AI Generated (Midjourney v6)** | `[x1: 95, y1: 110, x2: 360, y2: 390]` | **AI-GENERATED** | `98.65%` Fake |
| **AI Generated (StyleGAN2)** | `[x1: 60, y1: 52, x2: 280, y2: 275]` | **AI-GENERATED** | `99.88%` Fake |

---

## 📁 Repository Structure

```
Deep-fake-face-detection/
├── src/
│   ├── __init__.py          # Package exports
│   ├── face_detector.py      # MTCNN face detection & alignment module
│   ├── dataset.py            # Custom Dataset, JPEGCompression transform & augmentations
│   ├── model.py              # EfficientNet-B3 model architecture & checkpoint loader
│   ├── train.py              # Training, validation & evaluation routines
│   └── inference.py          # DeepFakePredictor pipeline with probability calibration
├── predict.py                # Standalone CLI entrypoint for local image classification
├── DeepFake_Face_Detection.ipynb  # Interactive Jupyter driver notebook importing src/
├── best_model_rebuilt/       # Serialized PyTorch model weights & checkpoint
├── requirements.txt          # Pinned dependency versions
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

---

## ⚙️ How to Run Locally

### 1. Clone Repository & Install Dependencies

```bash
git clone https://github.com/haseebarif11/Deep-fake-face-detection.git
cd Deep-fake-face-detection

pip install -r requirements.txt
```

### 2. Run Standalone CLI Inference

Classify a single image file:
```bash
python predict.py --image path/to/portrait.jpg
```

Classify an entire folder of images in bulk:
```bash
python predict.py --dir path/to/image_folder/
```

**CLI Command Options:**
- `--image` / `-i`: Path to single input image file.
- `--dir` / `-d`: Path to directory for batch classification.
- `--model` / `-m`: Path to custom model weights (default: `best_model_rebuilt`).
- `--threshold` / `-t`: Real-class probability threshold cutoff (default: `0.5`).
- `--no-face-detect`: Disable MTCNN face auto-cropping.

### 3. Launch Interactive Jupyter Notebook

To explore training routines, dataset augmentations, and evaluation visualizations:
```bash
jupyter notebook DeepFake_Face_Detection.ipynb
```

---

## 📌 Repository Metadata Reminders

> [!NOTE]
> Remember to set the repository description and topics in your **GitHub Repository Settings**:
> - **Description**: *A robust PyTorch EfficientNet-B3 system with MTCNN face alignment for AI-generated vs authentic face detection.*
> - **Topics**: `deepfake-detection`, `pytorch`, `computer-vision`, `deep-learning`, `face-detection`

---

<div align="center">
  Developed by <a href="https://github.com/haseebarif11">Haseeb Arif</a> • Released under the MIT License
</div>
