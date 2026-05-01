# Depth-Anything-V2: Monocular Depth Estimation

![Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2/raw/main/assets/teaser.png)

## 🎓 Academic Project Overview

This project is developed as part of the **Neural Networks and Deep Learning (NNDL)** and **Image Processing and Analysis** curriculum. It implements **Depth-Anything-V2**, a state-of-the-art model for monocular depth estimation.

**Group Members:**

- 👤 **Ahamed Najah**
- 👤 **Abdul Rahman Al tahir**

**Supervisor:**

- 👨‍🏫 **Engr. Faheem Ul Rehman Siddiqi**

---

## 🚀 Quick Start (Local Execution)

Follow these steps to get the project running on your local machine.

### 1. Environment Setup

We recommend using a virtual environment to avoid conflicts with global packages.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python app.py
```

Once running, open your browser and navigate to the local URL provided (usually `http://127.0.0.1:7860`).

---

## 🛠️ Features & Methodology

- **Model Architecture**: Utilizes Vision Transformer (ViT) backbones (Small, Base, Large) with a DPT head.
- **Dynamic Inference**: Automatically downloads pre-trained weights from Hugging Face Hub.
- **Interactive UI**: A Gradio-based interface allowing users to upload images and compare results using an interactive slider.
- **Output Formats**: Supports both colored depth maps and 16-bit raw disparity maps.

---

## ☁️ Deployment Guide

### Deploying to Hugging Face Spaces

1. Create a new **Space** on Hugging Face.
2. Select **Gradio** as the SDK.
3. Upload all files from this repository.
4. Ensure `requirements.txt` includes all necessary packages.
5. (Optional) For high performance, enable **ZeroGPU** in the Space settings.

---

## 📚 Acknowledgments

- Original Model by [Lihe Yang et al.](https://github.com/DepthAnything/Depth-Anything-V2)
- Built using **Gradio**, **PyTorch**, and **Hugging Face Hub**.
