"""
Renders OHLC data as candlestick chart images (PIL / matplotlib).

Main entry points
─────────────────
render_ohlc_to_pil(ohlc, img_size=224)
    Takes an (N, 4) numpy array [open, high, low, close] and returns a
    PIL.Image of the requested size.

generate_dataset(output_dir, n_per_class, n_points, img_size, val_split)
    Generates a full labeled synthetic dataset on disk:
        output_dir/
            train/
                head_and_shoulders/  img_00000.png …
                double_top/          …
                …
            val/
                …
"""

import io
import os
import random

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for Colab / scripts
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data_generation.patterns import (
    CLASS_NAMES,
    PATTERN_GENERATORS,
    generate_pattern_sample,
)


# ── Colour palette (matching common charting software) ─────────────────────────
_BULL_COLOUR  = "#26a69a"   # teal  — close ≥ open
_BEAR_COLOUR  = "#ef5350"   # red   — close <  open


def render_ohlc_to_pil(ohlc: np.ndarray, img_size: int = 224) -> Image.Image:
    """
    Render a single OHLC window as a candlestick chart.

    Args:
        ohlc:     np.array (N, 4) — [open, high, low, close]
        img_size: square output size in pixels

    Returns:
        PIL.Image (RGB, img_size × img_size)
    """
    n = len(ohlc)
    opens, highs, lows, closes = ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]

    inches = img_size / 100.0
    fig, ax = plt.subplots(figsize=(inches, inches), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    width = 0.65  # candle body width in x-axis units

    for i in range(n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        colour     = _BULL_COLOUR if c >= o else _BEAR_COLOUR

        # Wick
        ax.plot([i, i], [l, h], color=colour, linewidth=0.80, zorder=1, solid_capstyle="round")

        # Body
        body_lo = min(o, c)
        body_h  = max(abs(c - o), (h - l) * 0.01)   # minimum visible body
        rect    = mpatches.FancyBboxPatch(
            (i - width / 2, body_lo), width, body_h,
            boxstyle="square,pad=0",
            facecolor=colour, edgecolor=colour, linewidth=0.40, zorder=2,
        )
        ax.add_patch(rect)

    price_range = max(highs.max() - lows.min(), 1e-6)
    ax.set_xlim(-0.60, n - 0.40)
    ax.set_ylim(lows.min() - price_range * 0.04,
                highs.max() + price_range * 0.04)
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100,
                bbox_inches="tight", pad_inches=0, facecolor="white")
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize((img_size, img_size), Image.LANCZOS)
    plt.close(fig)
    return img


# ── Dataset generation ─────────────────────────────────────────────────────────

def generate_dataset(
    output_dir: str,
    n_per_class: int = 1500,
    n_points:    int = 60,
    img_size:    int = 224,
    val_split:   float = 0.15,
    seed:        int = 42,
) -> dict:
    """
    Generate a synthetic labelled candlestick image dataset.

    Creates the following on disk:
        output_dir/
            train/<class_name>/img_00000.png  …
            val/<class_name>/img_00000.png    …

    Args:
        output_dir:  root directory (created if absent)
        n_per_class: total samples per class (train + val combined)
        n_points:    candlesticks per chart window
        img_size:    output image size (square pixels)
        val_split:   fraction of samples used for validation
        seed:        random seed for reproducibility

    Returns:
        dict with keys 'train_counts' and 'val_counts'
    """
    np.random.seed(seed)
    random.seed(seed)

    n_val   = max(1, int(n_per_class * val_split))
    n_train = n_per_class - n_val

    counts = {"train": {}, "val": {}}

    for class_id, class_name in enumerate(CLASS_NAMES):
        for split, n_samples in [("train", n_train), ("val", n_val)]:
            split_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_dir, exist_ok=True)

            for idx in tqdm(
                range(n_samples),
                desc=f"{split}/{class_name}",
                leave=False,
            ):
                ohlc = generate_pattern_sample(class_id, n_points=n_points)
                img  = render_ohlc_to_pil(ohlc, img_size=img_size)
                img.save(os.path.join(split_dir, f"img_{idx:05d}.png"))

            counts[split][class_name] = n_samples

    total = sum(counts["train"].values()) + sum(counts["val"].values())
    print(f"\nDataset generated: {total} images total "
          f"({sum(counts['train'].values())} train / "
          f"{sum(counts['val'].values())} val)")
    return counts
