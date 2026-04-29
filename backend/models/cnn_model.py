"""
EfficientNet-B4 fine-tuned 3-class classifier:
  REAL (0) | AI_GENERATED (1) | EDITED (2)
Input: 380x380 RGB, ImageNet-normalized.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

CLASSES = ["REAL", "AI_GENERATED", "EDITED"]
NUM_CLASSES = 3


class FakeImageDetector(nn.Module):
    def __init__(self, pretrained: bool = True, num_classes: int = NUM_CLASSES):
        super().__init__()
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.efficientnet_b4(weights=weights)

        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )
        self.backbone = base
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # [B, num_classes] raw logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities [B, num_classes]."""
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=1)

    def predict_class(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (class_indices [B], class_probs [B, num_classes])."""
        probs = self.predict_proba(x)
        return probs.argmax(dim=1), probs

    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
