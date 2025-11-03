<!-- [![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.10%2B-orange)](https://www.tensorflow.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.patcog.2025.112150-blue)](https://ieeexplore.ieee.org/document/10884719#:~:text=10.1109/JSTARS.2025.3541733)
-->

[![Paper](https://img.shields.io/badge/Paper-IEEE%20JSTARS%202025-2b6cb0.svg)](https://ieeexplore.ieee.org/document/10884719#:~:text=10.1109/JSTARS.2025.3541733)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FJSTARS.2025.3541733-4a5568.svg)](10.1109/JSTARS.2025.3541733)
[![Tasks](https://img.shields.io/badge/Tasks-Classification-success.svg)](#-features)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-informational.svg)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange.svg)]()

## BrownViTNet: Hybrid CNN-Vision Transformer Model for the Classification of Brownfields in Aerial Imagery


### Abstract:
The identification of brownfield sites through satellite imagery is a crucial yet challenging classification problem in the field of remote sensing applications. In this study, we leverage aerial images sourced from Google Maps, Bing Maps, and national aerial and satellite imagery (DOP20), focusing on three distinct land use classes: active areas, construction sites, and brownfields, all matched across the same geographical coordinates. Our dataset initially includes images set against a 1000x1000 pixel blank canvas, often resulting in a significant portion of unused black space. To optimize this, images were cropped based on a threshold ensuring a minimum height and width of 400 pixels, resulting in a substantial reduction of the dataset by 42.8\% for active areas, 65.93\% for construction areas, and 75.86\% for brownfield images. Given the reduced size of our usable dataset, we employed single-image super-resolution to enhance image quality and effectively double our dataset size as a augmentation part of training data. For model architecture, the initial four layers consist of a Convolutional Neural Network (CNN), followed by intermediate layers using a Vision Transformer (ViT) with a patch size of 16. This novel hybrid architecture of BrownViTNet (Brownfield Vision Transformer Network) demonstrated impressive accuracy in the classification of brownfields from satellite imagery, presenting a significant advance in the use of machine learning for environmental monitoring and urban planning.

### Key Features

- Hybrid CNN–Vision Transformer Architecture — Combines convolutional feature extraction with transformer-based global attention.

- MSFE (Multistage Feature Extractor) — Depthwise and standard convolutions with a Channel Recalibration Unit for local feature refinement.

- HMSACTUnit (Hierarchical Multiscale Self-Attention Convolutional Transformer) — Fuses multi-head self-attention and convolutions to capture both local textures and global spatial context.

- Patch Embedding + Positional Encoding — ViT-style embedding to model inter-patch relationships and spatial hierarchy.

- Adaptive Learning Rate Scheduling — Evaluated cosine, linear, and exponential schedulers — exponential achieved the highest accuracy (~90.8%).

- Comprehensive Data Pipeline — Includes contour-based cropping, quality filtering, and Single-Image Super-Resolution (SISR) to enhance image clarity and double dataset size.

- Multi-Source Dataset Integration — Combines aerial imagery from Google Maps, Bing Maps, and DOP20 for diverse scene representation.

- Three-Class Classification — Differentiates among Active Industrial Sites, Brownfields, and Construction Areas with high precision.

- High Performance — Achieved up to 90.80% overall accuracy and AUC up to 0.97 on the Active class.

- Ablation & Comparative Studies — Show effectiveness of each architectural module and scheduler on model accuracy and generalization.

- Comprehensive Evaluation Metrics — Includes ROC–AUC, confusion matrix, precision, recall, and F1-score.

- Environment & Urban Development Relevance — Supports sustainable land monitoring and urban redevelopment assessment.


### 🧠 Model Architecture Overview

*Stage 1* – CNN Feature Extractor (MSFE):

- 4 convolutional blocks (Depthwise + Conv2D)

- BatchNorm + GELU activation

- Residual connections (shortcut + addition)

*Stage 2* – Transformer Encoder (HMSACTUnit):

- Patch extraction (Patches layer)

- Linear projection + positional embedding (PatchEncoder)

- 8 Transformer layers with:

- Multi-Head Self-Attention

- Feed-forward MLP (GELU + Dropout)

- Residual + LayerNorm connections

*Stage 3* – Classification Head:

- Flatten → Dense (2048, 1024)

- Dropout regularization

- Output: Dense(3) → softmax (via logits)




## 🧩 Repository Structure

```
BrownViTNet/
├── dataset_setup_aug.py      # Dataset loader and augmentation pipeline (JAX + TF-based)
├── helper_functions.py       # Utility functions (plotting, downloading, helpers)
├── model.py                  # BrownViTNet hybrid CNN–ViT architecture
├── train.py                  # Model training loop with warmup + scheduler + callbacks
├── requirements.txt          # Dependency list
├── LICENSE                   # MIT License
└── README.md                 # Project overview (this file)
```


## ⚙️ Training Configuration

This document outlines the hyperparameters, scheduler design, optimizer setup, and callback configuration used for training **BrownViTNet**.

---

### 🧠 Model Overview

- **Model Name:** BrownViTNet  
- **Architecture:** Hybrid CNN–Vision Transformer  
- **Input Shape:** (128 × 128 × 3)  
- **Output Classes:** 3 (Active, Construction, Brownfield)  
- **Base Framework:** TensorFlow 2.14 (Keras API)  
- **Mixed Precision:** Enabled (`mixed_float16`)  
- **GPU Acceleration:** TensorFlow + JAX-based augmentations  

---

### 🔧 Hyperparameters

| Parameter | Value | Description |
|------------|--------|-------------|
| `epochs` | **125** | Total number of training epochs |
| `batch_size` | **8** | Mini-batch size for training |
| `image_size` | **128 × 128** | Input image resolution |
| `val_split` | **0.2** | Validation split ratio |
| `label_smooth` | **0.05** | Label smoothing factor for loss regularization |
| `class_number` | **3** | Number of output categories |
| `autotune` | **tf.data.AUTOTUNE** | Prefetching for faster pipeline execution |
| `num_grad_accumulation` | **8** | Gradient accumulation steps |

---

### 🧩 Learning Rate & Scheduler

| Parameter | Value | Description |
|------------|--------|-------------|
| `lr_base` | **0.016** | Base learning rate |
| `lr_min` | **0.0** | Minimum learning rate floor |
| `lr_sched` | **cosine_restart** | Type of scheduler used |
| `lr_decay_epoch` | **2.4** | Epochs before each restart cycle |
| `lr_warmup_epoch` | **5** | Warmup period for learning rate ramp-up |
| `lr_decay_factor` | **0.97** | Exponential decay factor (if used) |
| `scaled_lr` | Computed as `lr_base * (batch_size / 256.0)` | Adaptive LR scaling |
| `scaled_lr_min` | Computed as `lr_min * (batch_size / 256.0)` | Adaptive LR minimum |
| `total_steps` | Derived from `epochs × steps_per_epoch` | Used for cosine scheduling |

**Learning Rate Policy:**  
Warm-up + Cosine Restart Decay, implemented via custom schedule class  
`WarmupLearningRateSchedule()` (see `train.py`).

---

### ⚙️ Optimizer & Loss Function

| Component | Configuration |
|------------|---------------|
| **Optimizer** | Adam (`amsgrad=True`) |
| **Loss Function** | Categorical Crossentropy (`from_logits=True`, `label_smoothing=0.05`) |
| **Metrics** | Accuracy, AUC, Precision, Recall, F1, PRC, TP, TN, FP, FN |

---

### ⏱️ Callbacks

| Callback | Description |
|-----------|-------------|
| `ModelCheckpoint` | Saves best model based on validation accuracy (`val_accuracy`) |
| `CSVLogger` | Logs per-epoch metrics into `training_log.csv` |
| `ReduceLROnPlateau` | Reduces LR when validation accuracy plateaus for 5 epochs |

**Model Save Path:**  
`CustomModel_3Chnl_mergeWithSuperImage.h5`

---

### 🧮 Dataset Parameters

| Parameter | Value | Description |
|------------|--------|-------------|
| **Data Source** | Aerial images (Google Maps, Bing Maps, DOP20) |
| **Train:Val Split** | 80:20 |
| **Data Augmentation** | Grayscale, channel shuffle, zoom, rotation |
| **Augmentation Framework** | Hybrid JAX + TensorFlow Sequential pipeline |
| **Prefetching** | Enabled (`AUTOTUNE`) for GPU efficiency |

---

### 🧾 Output Files

| File | Description |
|------|-------------|
| `CustomModel_3Chnl_mergeWithSuperImage.h5` | Saved best model weights |
| `training_log.csv` | Training history and validation metrics per epoch |
| `checkpoint/` (optional) | Stores checkpoint snapshots if configured |

---

### 📈 Expected Training Outcomes

| Metric | Typical Result |
|---------|----------------|
| **Training Accuracy** | ~91% |
| **Validation Accuracy** | ~90.8% |
| **AUC (Active class)** | 0.97 |
| **Loss (Final)** | ~0.23 |

---
