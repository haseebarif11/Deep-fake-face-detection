<div align="center">

# 🕵️‍♂️ DeepFake Face Detection

**A State-of-the-Art Deep Learning Model built with PyTorch to detect AI-generated faces.**

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue?style=for-the-badge)](https://huggingface.co/spaces/HaseebArif11/DeepFake-Face-Detection)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)

<img src="https://cdn-thumbnails.huggingface.co/social-thumbnails/spaces/HaseebArif11/DeepFake-Face-Detection.png" alt="DeepFake Face Detection" width="600" style="border-radius:10px; box-shadow: 0px 4px 15px rgba(0,0,0,0.2); margin-top: 20px;">

</div>

---

## 🚀 Live Demo

Experience the model in action directly from your browser!

👉 **[Launch DeepFake Face Detection on Hugging Face](https://huggingface.co/spaces/HaseebArif11/DeepFake-Face-Detection)**

Upload any portrait photo, and the tool will instantly analyze it to determine whether it is an **Authentic** human face or **AI-Generated**. It provides a clear, high-confidence verdict in seconds.

---

## 🧠 Model Overview

The core of this project is a Deep Convolutional Neural Network (CNN) designed to identify the subtle, almost imperceptible artifacts left behind by modern AI generation methods (like GANs and Diffusion models). 

By extracting deep hierarchical features, the model learns the structural difference between real human skin/features and synthesized pixels.

---

## 📁 Project Structure

| File / Folder | Description |
| :--- | :--- |
| `DeepFake_Face_Detection.ipynb` | The core Jupyter Notebook containing data preprocessing, model architecture, training loops, and evaluation metrics. |
| `best_model_rebuilt/` | Contains the serialized PyTorch model weights and architecture, ready for inference. |

---

## 🛠️ Technologies & Libraries

- **Deep Learning:** `PyTorch`, `TorchVision`
- **Data Engineering:** `NumPy`, `Pandas`
- **Computer Vision:** `PIL (Pillow)`
- **Data Visualization:** `Matplotlib`
- **Model Evaluation:** `Scikit-learn`

---

## ⚙️ How to Run Locally

To explore the code or run inference on your own machine:

**1. Clone the repository**
```bash
git clone https://github.com/haseebarif11/Deep-fake-face-detection.git
cd Deep-fake-face-detection
```

**2. Install dependencies**
```bash
pip install torch torchvision numpy pandas matplotlib pillow scikit-learn
```

**3. Explore the Notebook**
Launch Jupyter to explore the training process:
```bash
jupyter notebook DeepFake_Face_Detection.ipynb
```

---

<div align="center">
  <i>Developed by <a href="https://github.com/haseebarif11">Haseeb Arif</a></i>
</div>
