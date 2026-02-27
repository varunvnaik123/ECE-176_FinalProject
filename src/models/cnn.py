"""
Custom 3-layer CNN for chart pattern classification.

Architecture mirrors the ECE 176 course progression:
  Conv2D  (3×3 kernels, hierarchical features)   — Lecture 5
  BatchNorm                                        — Lecture 7
  ReLU activations                                 — Lecture 4
  MaxPooling (translation invariance)              — Lecture 5
  Dropout (p=0.5, overfitting prevention)          — Lecture 7
  Softmax + Cross-Entropy (7-class output)         — Lecture 4

Weight init: Kaiming (He) for conv layers, Xavier for linear layers — Lecture 7.

Usage
─────
    from src.models.cnn import PatternCNN
    model = PatternCNN(num_classes=7)
    logits = model(imgs)          # (B, 7)
    cam_features = model.get_feature_maps(imgs)   # (B, 128, 28, 28) for Grad-CAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    """Conv → BN → ReLU → MaxPool block."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class PatternCNN(nn.Module):
    """
    Three-block CNN → Global Average Pool → two-layer classifier.

    Input:  (B, 3, 224, 224) normalised RGB chart image
    Output: (B, num_classes)  raw logits (apply softmax for probabilities)

    Channel progression:  3 → 32 → 64 → 128
    Spatial progression:  224 → 112 → 56 → 28  (after three MaxPool2d)
    """

    def __init__(self, num_classes: int = 7, dropout_p: float = 0.50):
        super().__init__()

        # ── Feature extractor ─────────────────────────────────────────────────
        self.block1 = _ConvBlock(3,   32, pool=True)   # 224 → 112
        self.block2 = _ConvBlock(32,  64, pool=True)   # 112 → 56
        self.block3 = _ConvBlock(64, 128, pool=True)   # 56  → 28

        # Global Average Pool: (B, 128, 28, 28) → (B, 128, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Classifier ────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    # ── Weight initialisation (Lecture 7) ─────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward pass ──────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = x.flatten(1)
        return self.classifier(x)

    # ── Grad-CAM support ──────────────────────────────────────────────────────
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the activation map BEFORE global average pooling.
        Used by GradCAM to compute spatial attention maps.

        Returns: (B, 128, 28, 28)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def get_classifier_weights(self) -> torch.Tensor:
        """
        Return the weight matrix of the final linear layer.
        Shape: (num_classes, 64)  — used by CAM (not Grad-CAM).
        """
        return list(self.classifier.children())[-1].weight

    # ── Convenience ───────────────────────────────────────────────────────────
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax-normalised class probabilities. Shape: (B, num_classes)."""
        return F.softmax(self.forward(x), dim=1)
