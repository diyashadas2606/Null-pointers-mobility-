# Real-Time Drivable Space Segmentation
### MIT Hackathon — AI and Computer Vision Challenge | Track: Problem Statement 2

---

## Project Overview

This project implements an end-to-end pipeline for **real-time drivable space segmentation** using the **nuScenes v1.0-mini** dataset. The system takes raw camera images from a multi-camera autonomous vehicle rig, projects HD-map drivable-area information into camera-view binary masks, and trains a residual U-Net to predict where the vehicle can safely drive.

The pipeline is designed to run on **Google Colab (T4/L4 GPU, ~15 GB VRAM, ~16 GB RAM)** with careful memory management throughout.

---

## Model Architecture

The model is a custom **ResUNetBEV** — a U-Net with residual (skip-connection) double-convolution blocks at every stage.

```
Input Image (3 × 256 × 704)
        │
  DoubleConv (32)          ← Encoder begins
        │
  Down1 → MaxPool + DoubleConv (64)
  Down2 → MaxPool + DoubleConv (128)
  Down3 → MaxPool + DoubleConv (256)
  Down4 → MaxPool + DoubleConv (512)   ← Bottleneck
        │
  Up1 → ConvTranspose + DoubleConv (256)   ← Decoder begins
  Up2 → ConvTranspose + DoubleConv (128)
  Up3 → ConvTranspose + DoubleConv (64)
  Up4 → ConvTranspose + DoubleConv (32)
        │
  Output Conv (1 × 256 × 704)           ← Binary drivable mask
```

**Key design choices:**
- Each `DoubleConv` block has a residual shortcut (1×1 conv if channels differ), combining U-Net's spatial recovery with ResNet's stable gradient flow.
- `ConvTranspose2d` for learned upsampling (not bilinear interpolation) in the decoder path.
- Single-channel sigmoid output for binary segmentation.

**Loss function — Composite (weighted sum):**
| Component | Weight | Purpose |
|-----------|--------|---------|
| Binary Cross-Entropy | 0.55 | Overall pixel-level accuracy |
| Dice Loss | 0.30 | Handles class imbalance (road vs. non-road) |
| Boundary Loss (Sobel) | 0.15 | Sharper drivable-area edge prediction |

---

## Dataset Used

**nuScenes v1.0-mini** — a public autonomous driving dataset by Motional.

| Property | Value |
|----------|-------|
| Total scenes | 10 |
| Train scenes | 8 (`scene-0061`, `scene-0553`, `scene-0655`, `scene-0757`, `scene-0796`, `scene-1077`, `scene-1094`, `scene-1100`) |
| Val scenes | 2 (`scene-0103`, `scene-0916`) |
| Cameras per sample | 6 (FRONT, FRONT\_LEFT, FRONT\_RIGHT, BACK, BACK\_LEFT, BACK\_RIGHT) |
| Raw image resolution | 1600 × 900 |
| Processed resolution | 704 × 256 |
| Annotation format | HD-map drivable area projected to camera view |

The split is performed at the **scene level** (not frame level) to prevent temporal data leakage between adjacent 2Hz keyframes.

---

## Preprocessing Pipeline

The full offline preprocessing pipeline runs once and writes `.pkl` cache files for fast DataLoader access during training.

```
Stage 1 │ JSON Metadata Indexing  — O(1) token→record dictionaries
Stage 2 │ Scene-Level Splitting   — 8 train / 2 val scenes (no leakage)
Stage 3 │ Drivable Mask Gen       — HD-map → camera-view binary masks
Stage 4 │ Image Preprocessing     — resize 1600×900 → 704×256, scale K
Stage 5 │ Augmentation Pipeline   — flip, scale/crop, HSDA (FFT)
Stage 6 │ CBGS Sampling           — class-balanced grouping & sampling
Stage 7 │ PKL Serialization       — flat offline cache for fast loading
Stage 8 │ PyTorch Dataset/Loader  — memory-safe, AMP-ready DataLoader
```

**Mask generation** uses a three-stage coordinate transform:
1. Sample a dense BEV grid (ground plane, z=0) in ego frame
2. Transform to global frame → query HD-map PNG for drivable pixels
3. Transform drivable points → camera frame → project to image via scaled intrinsic K

**Augmentations (training only):**
| Augmentation | Parameter |
|---|---|
| Random Horizontal Flip | p = 0.50 |
| Random Scale + Crop | scale ∈ [0.94, 1.11] |
| HSDA (Fourier high-freq shuffle) | p = 0.50, α = 0.15 |

Every spatial transform is mirrored in the intrinsic matrix **K** to preserve pixel↔3D geometric consistency.

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (T4/L4 recommended for Colab)
- Google Drive mounted (for Colab use)

### Install dependencies

```bash
pip install nuscenes-devkit opencv-python-headless numpy pillow torch torchvision
```

Or in Colab:
```python
!pip install -q nuscenes-devkit opencv-python-headless
```

### Dataset setup

1. Download [nuScenes v1.0-mini](https://www.nuscenes.org/download) and extract it.
2. Ensure the following directory structure:
```
Extracted_data/
├── v1.0-mini/
│   ├── scene.json
│   ├── sample.json
│   ├── sample_data.json
│   ├── calibrated_sensor.json
│   ├── ego_pose.json
│   ├── map.json
│   └── log.json
├── maps/
│   └── *.png   (HD map PNGs)
└── samples/
    ├── CAM_FRONT/
    ├── CAM_FRONT_LEFT/
    └── ...
```

---

## How to Run the Code

### Step 1 — Preprocessing (run once)

```python
from nuscenes_preprocessing import PreprocessingConfig, run_preprocessing

cfg = PreprocessingConfig(
    data_root  = "/content/Extracted_data",   # path to dataset
    output_dir = "/content/preprocessed",      # output for .pkl files
)
train_loader, val_loader = run_preprocessing(cfg)
```

This writes `train_metadata.pkl` and `val_metadata.pkl` to `output_dir`.

### Step 2 — Training

```python
from Mobility_Model import CFG, run_training

cfg = CFG(
    train_pkl = "/content/preprocessed/train_metadata.pkl",
    val_pkl   = "/content/preprocessed/val_metadata.pkl",
    out_dir   = "/content/checkpoints",
    epochs    = 25,
    batch_size = 4,
    lr         = 1e-3,
)
best_ckpt = run_training(cfg)
```

### Step 3 — Testing / Inference

```python
from Mobility_Model import CFG, run_testing

metrics = run_testing(cfg, best_ckpt, save_predictions_dir="/content/predictions")
print(metrics)
# Example: {"mIoU": 0.82, "Dice": 0.89, "PixelAcc": 0.96, "FPS": 48.3}
```

### Training configuration reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 25 | Training epochs |
| `batch_size` | 4 | Mini-batch size |
| `lr` | 1e-3 | Peak learning rate (OneCycleLR) |
| `weight_decay` | 1e-4 | AdamW regularisation |
| `amp` | True | FP16 mixed precision |
| `accum_steps` | 2 | Gradient accumulation steps |
| `threshold` | 0.5 | Sigmoid binarisation threshold |

---

## Example Outputs / Results

The model outputs a binary mask of shape `(H, W)` for each input image, where `1 = drivable` and `0 = non-drivable`.

**Evaluation metrics tracked:**

| Metric | Description |
|--------|-------------|
| **mIoU** | Mean Intersection over Union (primary metric) |
| **Dice Score** | Harmonic mean of precision and recall over pixels |
| **Pixel Accuracy** | Fraction of correctly classified pixels |
| **FPS** | Inference frames per second on T4 GPU |

Training logs are printed per epoch:
```
Epoch 01/25 | train_loss=0.4821 | val_mIoU=0.6134 | val_Dice=0.7428 | val_Acc=0.9012 | FPS=51.3 | 2.4 min
Epoch 10/25 | train_loss=0.2103 | val_mIoU=0.7891 | val_Dice=0.8734 | val_Acc=0.9501 | FPS=52.1 | 2.3 min
Epoch 25/25 | train_loss=0.1544 | val_mIoU=0.8412 | val_Dice=0.9021 | val_Acc=0.9678 | FPS=52.8 | 2.2 min
```

---

## Project Structure

```
├── nuscenes_preprocessing.py   # Full offline preprocessing pipeline (Stages 1–8)
├── Mobility_Model.ipynb        # ResUNetBEV model, training, and evaluation
└── README.md
```

---

## Key References

- nuScenes Dataset — Caesar et al., CVPR 2020
- BEVDet — Huang et al., arXiv:2112.11790
- HSDA — Glisson et al., WACV 2025 (arXiv:2412.06127)
- CBGS Sampling — Zhu et al., arXiv:2203.17054
- Image downscaling strategy — arXiv:2312.00633

---

## Declaration

This work is original and developed entirely by the team. All team members agree to the hackathon rules and evaluation process.
