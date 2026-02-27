"""
Grad-CAM (Gradient-weighted Class Activation Maps) implementation.

Reference: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization", ICCV.

Algorithm (Lecture 11 — visualising deep networks):
  1. Forward pass  → save the last convolutional activation map A^k  ∈ R^{h×w}
  2. Backward pass for the target class score Y^c
     → save dY^c/dA^k for every channel k
  3. Global-average-pool the gradients:
        α_k = (1/Z) Σ_{i,j} ∂Y^c/∂A^k_{ij}
  4. Weighted combination + ReLU:
        L^c_{GradCAM} = ReLU( Σ_k α_k · A^k )
  5. Bilinear upsample to input resolution.

Usage
─────
    from src.visualization.gradcam import GradCAM, visualize_gradcam

    # Attach to the last conv block of PatternCNN
    cam = GradCAM(model, target_layer=model.block3[0])   # first Conv2d in block3

    heatmap, pred, conf = cam.compute(img_tensor)        # img_tensor: (1,3,H,W)
    fig = visualize_gradcam(original_pil, heatmap, pred, conf)
"""

from typing import Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Grad-CAM for any PyTorch model with at least one convolutional layer.

    Args:
        model:        nn.Module in eval mode
        target_layer: the convolutional layer whose activations we hook into
                      (typically the last conv layer for best spatial resolution)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients   = None

        # Forward hook — saves feature maps during forward pass
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        # Backward hook — saves gradients during backward pass
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    # ── Hooks ─────────────────────────────────────────────────────────────────

    def _save_activation(self, _module, _inp, output):
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def remove_hooks(self):
        """Call after use to avoid memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    # ── Core computation ──────────────────────────────────────────────────────

    def compute(
        self,
        x:          torch.Tensor,
        class_idx:  Optional[int] = None,
    ) -> tuple[np.ndarray, int, float]:
        """
        Compute the Grad-CAM heatmap for one image.

        Args:
            x:         input tensor (1, C, H, W)  — batched with batch size 1
            class_idx: target class index; if None, use the predicted class

        Returns:
            heatmap:   np.array (H, W) in [0, 1], upsampled to input resolution
            pred_class: int — predicted class index
            confidence: float — softmax probability of the predicted class
        """
        self.model.eval()
        self.model.zero_grad()

        x = x.requires_grad_(True)

        # ── Forward ───────────────────────────────────────────────────────────
        logits  = self.model(x)
        probs   = F.softmax(logits, dim=1)

        pred_class = int(logits.argmax(dim=1).item())
        confidence = float(probs[0, pred_class].item())
        target     = class_idx if class_idx is not None else pred_class

        # ── Backward for target class ──────────────────────────────────────────
        score = logits[0, target]
        score.backward()

        # ── Grad-CAM formula ──────────────────────────────────────────────────
        # α_k = global-average-pool of ∂Y^c/∂A^k
        alpha = self._gradients.mean(dim=[2, 3], keepdim=True)  # (1, K, 1, 1)

        # Weighted sum + ReLU
        cam = (alpha * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam_np = cam.squeeze().cpu().numpy()
        if cam_np.max() > 0:
            cam_np /= cam_np.max()

        # Upsample to input spatial resolution
        cam_t = torch.FloatTensor(cam_np).unsqueeze(0).unsqueeze(0)
        cam_up = F.interpolate(
            cam_t,
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()

        return cam_up, pred_class, confidence

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()


# ── Visualisation helper ───────────────────────────────────────────────────────

def visualize_gradcam(
    original_img: Image.Image,
    heatmap:      np.ndarray,
    pred_class:   int,
    confidence:   float,
    class_names:  Optional[list] = None,
    save_path:    Optional[str]  = None,
    alpha:        float          = 0.45,
) -> plt.Figure:
    """
    Create a three-panel Grad-CAM visualisation:
        [original] | [heatmap] | [overlay]

    Args:
        original_img: PIL.Image (RGB)
        heatmap:      np.array (H, W) in [0, 1]
        pred_class:   predicted class index
        confidence:   softmax confidence
        class_names:  list of string labels (optional)
        save_path:    if given, save figure to this path
        alpha:        blend weight for heatmap overlay (0=original, 1=heatmap)

    Returns:
        matplotlib Figure
    """
    label = class_names[pred_class] if class_names else str(pred_class)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel 1 — original
    axes[0].imshow(original_img)
    axes[0].set_title("Original Chart", fontsize=12)
    axes[0].axis("off")

    # Panel 2 — raw heatmap
    im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3 — overlay
    orig_arr    = np.array(original_img.resize(
        (heatmap.shape[1], heatmap.shape[0]), Image.LANCZOS
    ), dtype=float) / 255.0
    heat_rgb    = cm.jet(heatmap)[:, :, :3]
    overlay     = np.clip((1 - alpha) * orig_arr + alpha * heat_rgb, 0, 1)

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Overlay\nPred: {label}  ({confidence:.1%})",
        fontsize=12,
    )
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ── Batch Grad-CAM ────────────────────────────────────────────────────────────

def gradcam_grid(
    model,
    target_layer,
    images:      list,     # list of PIL.Image
    transform,
    device,
    class_names: Optional[list] = None,
    n_cols:      int = 4,
    save_path:   Optional[str]  = None,
) -> plt.Figure:
    """
    Generate a grid of Grad-CAM overlays for a list of images.

    Each column shows one image; each row shows: original / overlay.

    Args:
        model:        trained PyTorch model (eval mode)
        target_layer: convolutional layer to hook
        images:       list of PIL.Image (RGB)
        transform:    val transform pipeline matching training preprocessing
        device:       torch.device
        class_names:  optional label list
        n_cols:       images per row
        save_path:    optional output path

    Returns:
        matplotlib Figure
    """
    n    = len(images)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = axes.reshape(2, 1)

    with GradCAM(model, target_layer) as gcam:
        for col, pil_img in enumerate(images):
            tensor  = transform(pil_img).unsqueeze(0).to(device)
            heatmap, pred, conf = gcam.compute(tensor)

            label   = class_names[pred] if class_names else str(pred)
            orig_np = np.array(pil_img)
            heat_rgb= cm.jet(heatmap)[:, :, :3]
            orig_f  = orig_np.astype(float) / 255.0
            overlay = np.clip(0.55 * orig_f + 0.45 * heat_rgb, 0, 1)

            axes[0, col].imshow(pil_img)
            axes[0, col].set_title(f"{label}\n{conf:.1%}", fontsize=9)
            axes[0, col].axis("off")

            axes[1, col].imshow(overlay)
            axes[1, col].set_title("Grad-CAM", fontsize=9)
            axes[1, col].axis("off")

    plt.suptitle("Grad-CAM — CNN attention per detected pattern", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
