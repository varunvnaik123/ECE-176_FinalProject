"""
ResNet-18 with ImageNet transfer learning for chart pattern classification.

Transfer-learning strategy (Lecture 9):
  1. Load ResNet-18 pretrained on ImageNet.
  2. Replace the final FC head (1000 → num_classes).
  3. Two fine-tuning modes:
       frozen_backbone=True  → only train the new head  (fast, few labelled samples)
       frozen_backbone=False → train all layers          (better with more data)

Usage
─────
    from src.models.resnet18 import get_resnet18
    model = get_resnet18(num_classes=7, pretrained=True, frozen_backbone=False)
"""

import torch
import torch.nn as nn
from torchvision import models


def get_resnet18(
    num_classes:     int  = 7,
    pretrained:      bool = True,
    frozen_backbone: bool = False,
    dropout_p:       float = 0.30,
) -> nn.Module:
    """
    Build a ResNet-18 adapted for chart-pattern classification.

    Args:
        num_classes:     output classes (7 for this project)
        pretrained:      load ImageNet weights via torchvision
        frozen_backbone: freeze all layers except the final FC head
        dropout_p:       dropout before the classification head

    Returns:
        nn.Module ready for training / inference
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.resnet18(weights=weights)

    # Optionally freeze the backbone
    if frozen_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    # Replace the classification head
    in_features = model.fc.in_features          # 512 for ResNet-18
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_p),
        nn.Linear(256, num_classes),
    )

    # Kaiming init for the new head
    for m in model.fc.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

    return model


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters (call after initial head-only training)."""
    for param in model.parameters():
        param.requires_grad = True


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
