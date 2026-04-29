"""
Dataset loader for fake image detection (3-class).

Expected directory layout:
    data/
      train/
        real/          *.jpg / *.png / *.jpeg / *.webp
        ai_generated/  *.jpg / *.png / *.jpeg / *.webp
        edited/        *.jpg / *.png / *.jpeg / *.webp
      val/
        real/
        ai_generated/
        edited/

Label mapping: real=0, ai_generated=1, edited=2

Compatible sources:
  real:          FFHQ, Flickr-Faces (140k dataset real split)
  ai_generated:  StyleGAN faces (140k fake split), DALL-E, Stable Diffusion generated
  edited:        CASIA v2, Coverage, CoMoFoD (copy-move + splicing + retouching)
"""

import io
import random
from pathlib import Path
from typing import Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as T
import numpy as np

IMG_SIZE = 380  # EfficientNet-B4 native resolution
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CLASS_NAMES = ["real", "ai_generated", "edited"]
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class JPEGAugment:
    """Simulate JPEG compression — hardens model against compression artifacts."""
    def __init__(self, quality_range=(60, 95)):
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        q = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class GaussianNoise:
    def __init__(self, std_range=(0.0, 0.02)):
        self.std_range = std_range

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        std = random.uniform(*self.std_range)
        return (tensor + torch.randn_like(tensor) * std).clamp(0, 1)


def build_transforms(split: str) -> T.Compose:
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    if split == "train":
        return T.Compose([
            T.Resize((IMG_SIZE + 40, IMG_SIZE + 40)),
            T.RandomCrop(IMG_SIZE),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=12),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomGrayscale(p=0.05),
            JPEGAugment(quality_range=(55, 95)),
            T.ToTensor(),
            GaussianNoise(std_range=(0.0, 0.015)),
            normalize,
        ])
    else:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            normalize,
        ])


class FakeImageDataset(Dataset):
    def __init__(self, root: str, split: str = "train"):
        self.split = split
        base = Path(root) / split

        self.paths: list[Tuple[Path, int]] = []
        counts = {}

        for class_name, label in LABEL_MAP.items():
            class_dir = base / class_name
            if not class_dir.exists():
                print(f"  WARNING: {class_dir} not found, skipping")
                counts[class_name] = 0
                continue
            imgs = [p for p in class_dir.rglob("*") if p.suffix.lower() in EXTENSIONS]
            self.paths.extend([(p, label) for p in imgs])
            counts[class_name] = len(imgs)

        if not self.paths:
            raise FileNotFoundError(
                f"No images found under {base}. "
                "Expected subdirs: real/, ai_generated/, edited/"
            )

        self.transform = build_transforms(split)

        total = len(self.paths)
        print(f"[{split}] " + "  ".join(f"{k}={v:,}" for k, v in counts.items()) + f"  total={total:,}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(128, 128, 128))
        return self.transform(img), torch.tensor(label, dtype=torch.long)

    def make_sampler(self) -> WeightedRandomSampler:
        """Balanced sampler: each class gets equal expected frequency per batch."""
        labels = [lbl for _, lbl in self.paths]
        num_classes = len(CLASS_NAMES)
        class_counts = [labels.count(c) for c in range(num_classes)]
        class_weights = [1.0 / max(c, 1) for c in class_counts]
        sample_weights = [class_weights[lbl] for lbl in labels]
        return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
