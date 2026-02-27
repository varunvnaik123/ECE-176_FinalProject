"""
PyTorch Dataset and DataLoader utilities for chart-pattern image data.

Directory layout expected on disk
──────────────────────────────────
    root/
        train/
            head_and_shoulders/    img_00000.png  img_00001.png  …
            double_top/            …
            descending_triangle/   …
            inv_head_and_shoulders/…
            double_bottom/         …
            ascending_triangle/    …
            no_pattern/            …
        val/
            <same structure>

Augmentation strategy (Lecture 7)
──────────────────────────────────
    Train: random horizontal flip (patterns are NOT strictly direction-invariant
           so we keep this at p=0.25), minor colour jitter (brightness/contrast),
           resize to img_size.
    Val:   only resize + normalise.

ImageNet mean/std normalisation is used so both our CNN and ResNet-18 receive
the same preprocessing (important for fair comparison and for transfer learning).
"""

import os
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


# ── Label mapping ──────────────────────────────────────────────────────────────

CLASS_TO_IDX = {
    "head_and_shoulders":     0,
    "double_top":             1,
    "descending_triangle":    2,
    "inv_head_and_shoulders": 3,
    "double_bottom":          4,
    "ascending_triangle":     5,
    "no_pattern":             6,
}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# Human-readable short labels for plots
SHORT_NAMES = ["H&S", "DblTop", "DescTri", "InvH&S", "DblBot", "AscTri", "NoPat"]


# ── Dataset ────────────────────────────────────────────────────────────────────

class ChartPatternDataset(Dataset):
    """
    Reads candlestick chart PNG images from a directory tree and returns
    (image_tensor, class_index) pairs.
    """

    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir:  path to split directory (e.g. data/synthetic/train)
            transform: torchvision transform pipeline
        """
        self.root_dir  = root_dir
        self.transform = transform
        self.samples: list[Tuple[str, int]] = []

        for class_name, class_idx in CLASS_TO_IDX.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(".png"):
                    self.samples.append((os.path.join(class_dir, fname), class_idx))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No PNG images found in {root_dir}. "
                "Run notebook 01_generate_synthetic_data.ipynb first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_labels(self):
        return [lbl for _, lbl in self.samples]

    def class_counts(self) -> dict:
        counts = {}
        for _, lbl in self.samples:
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts


# ── Transforms ────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(train: bool = True, img_size: int = 224):
    """
    Return the appropriate torchvision transform pipeline.

    Train augmentations (Lecture 7):
        - RandomHorizontalFlip (p=0.25): mild augmentation (not all patterns are
          horizontally symmetric, so keep probability low)
        - ColorJitter: brightness/contrast variation to improve real-data generalisation
        - Resize + ToTensor + Normalise

    Val / test: only Resize + ToTensor + Normalise.
    """
    normalise = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.25),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
            transforms.ToTensor(),
            normalise,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalise,
        ])


# ── DataLoaders ───────────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir:    str,
    batch_size:  int = 32,
    img_size:    int = 224,
    num_workers: int = 2,
    balanced:    bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.

    Args:
        data_dir:    root data directory containing train/ and val/ subdirs
        batch_size:  mini-batch size (Lecture 6 — affects gradient noise + BN stats)
        img_size:    image resize target (224 for ResNet compatibility)
        num_workers: parallel data-loading workers
        balanced:    if True, use WeightedRandomSampler for class balance

    Returns:
        (train_loader, val_loader)
    """
    train_ds = ChartPatternDataset(
        os.path.join(data_dir, "train"),
        transform=get_transforms(train=True,  img_size=img_size),
    )
    val_ds = ChartPatternDataset(
        os.path.join(data_dir, "val"),
        transform=get_transforms(train=False, img_size=img_size),
    )

    # Optional balanced sampler (useful when no_pattern dominates real data)
    if balanced:
        labels  = train_ds.get_labels()
        counts  = torch.bincount(torch.tensor(labels), minlength=7).float()
        weights = 1.0 / (counts + 1e-6)
        sample_weights = weights[torch.tensor(labels)]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train: {len(train_ds):,} images  |  Val: {len(val_ds):,} images")
    print(f"Train class counts: {train_ds.class_counts()}")
    return train_loader, val_loader
