# Fake Image Detection Dataset

3-class dataset for FakeGuard AI training.

## Classes
| Label | Folder | Description |
|-------|--------|-------------|
| 0 | `real/` | Authentic camera photographs (Flickr-Faces) |
| 1 | `ai_generated/` | GAN / Stable Diffusion generated images (StyleGAN) |
| 2 | `edited/` | Copy-move, splicing, retouching (CASIA v2) |

## Structure
```
train/
  real/           (8000 images)
  ai_generated/   (8000 images)
  edited/         (8000 images)
val/
  real/           (2000 images)
  ai_generated/   (2000 images)
  edited/         (2000 images)
```

## Usage
Populated via Colab training notebook:
https://github.com/RuudyLinux/Fake-Image-Detections/blob/main/colab_fake_image_detection.ipynb

## Model
EfficientNet-B4, 3-class CrossEntropyLoss, 30 epochs T4 GPU.
Expected macro-F1: 0.85–0.93
