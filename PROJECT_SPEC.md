# Fake Image Detection System - Project Specification

## Goal
Build a dual-approach, multi-modal system to detect fake images with explainable output.

## Detection Modes
- Blind detection: single image, no original reference.
- Comparative detection: original + suspected image, localize manipulation.

## Core AI Modules
- CNN or vision model for artifact detection.
- Frequency analysis (FFT or DCT) for hidden patterns.
- Noise residual or GAN fingerprint detection.
- Comparative module: SSIM + pixel difference.
- Fusion engine: combine module outputs.

## Outputs
- Prediction: Real, AI Generated, or Edited.
- Confidence score.
- Reason: artifact, frequency anomaly, or fingerprint.
- Heatmap visualization (explainable AI).

## Data and Training
- Classes: real, ai_generated, edited.
- Dataset size: minimum 1000 images per class; recommended 3000 to 5000.
- Must be balanced.
- Structure:

```
dataset/
  train/
    real/
    ai_generated/
    edited/
  val/
    real/
    ai_generated/
    edited/
```

## Training Setup
- Platform: Google Colab with T4 GPU.
- Framework: PyTorch.
- Model: ResNet50 or EfficientNet (transfer learning).
- Preprocessing: resize 224x224, normalize.
- Augmentation: flip, rotation, brightness, compression.
- Loss: CrossEntropyLoss.
- Optimizer: Adam, lr=0.0001.
- Epochs: 10 to 30.

## Evaluation
- Accuracy, Precision, Recall, F1 score.

## Website Requirements
### Frontend
- Modern UI, dark theme, purple accent.
- Two options: Detect Without Original, Compare With Original.
- Drag and drop image upload.
- Result dashboard with heatmap and analysis.

### Backend
- FastAPI.
- Modular pipeline for artifact, frequency, fingerprint, and comparison.

## Phase 2 Features
- Grad-CAM heatmaps.
- FFT-based model integration.
- GAN fingerprint detection.
- Fusion scoring system.
- PDF report generation.
- History dashboard.

## Development Plan
1. Dataset setup and preprocessing.
2. Basic model training (CNN).
3. Website UI and backend setup.
4. Model integration.
5. Advanced features: FFT, heatmap, fusion.

## Immediate Next Steps
- Prepare dataset on Google Drive and validate structure.
- Run baseline Colab training notebook.
