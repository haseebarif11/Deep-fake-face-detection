# 🕵️‍♂️ DeepFake Face Detection

![DeepFake Face Detection](https://cdn-thumbnails.huggingface.co/social-thumbnails/spaces/HaseebArif11/DeepFake-Face-Detection.png)

A deep learning project built with PyTorch to detect whether a face image is authentic or AI-generated (DeepFake).

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/HaseebArif11/DeepFake-Face-Detection)

## 🚀 Live Demo

Try out the live web application here: [**DeepFake Face Detection on Hugging Face**](https://huggingface.co/spaces/HaseebArif11/DeepFake-Face-Detection)

Upload a photo, and the tool will analyze it to tell whether it looks authentic or was created by AI. It provides a clear verdict—"AI-Generated" or "Authentic"—along with a confidence percentage.

## 📁 Project Structure

- `DeepFake_Face_Detection.ipynb`: The main Jupyter Notebook containing the data preprocessing, model architecture, training, and evaluation code.
- `best_model_rebuilt/`: The saved PyTorch model weights.

## 🛠️ Technologies Used

- **Frameworks:** PyTorch, TorchVision
- **Data Manipulation:** NumPy, Pandas
- **Image Processing:** PIL (Pillow)
- **Visualization:** Matplotlib
- **Machine Learning Metrics:** Scikit-learn

## 🧠 Model Overview

The model uses a deep convolutional neural network (CNN) architecture trained to distinguish the subtle artifacts left by AI generation methods compared to authentic human faces.

## ⚙️ How to Use Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/HaseebArif11/DeepFake-Face-Detection.git
   cd DeepFake-Face-Detection
   ```
2. Install the required dependencies:
   ```bash
   pip install torch torchvision numpy pandas matplotlib pillow scikit-learn
   ```
3. Open `DeepFake_Face_Detection.ipynb` in Jupyter Notebook or Jupyter Lab to explore the training process and evaluation metrics.
