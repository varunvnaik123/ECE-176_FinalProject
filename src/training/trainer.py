"""
Training loop for chart-pattern CNNs.

Implements:
  • Adam optimiser with step-decay learning-rate schedule  (Lecture 6)
  • Gradient clipping                                      (training stability)
  • Per-epoch validation                                   (Lecture 3 — model selection)
  • Early stopping                                         (Lecture 7)
  • Best-model checkpointing

Usage
─────
    from src.training.trainer import Trainer

    trainer = Trainer(model, train_loader, val_loader,
                      lr=1e-3, weight_decay=1e-4, device="cuda")
    history = trainer.fit(n_epochs=40, patience=8, save_path="checkpoints/best.pt")
"""

import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix


class Trainer:
    """
    Generic trainer for classification models.

    Args:
        model:         PyTorch nn.Module
        train_loader:  training DataLoader
        val_loader:    validation DataLoader
        lr:            initial Adam learning rate
        weight_decay:  L2 regularisation coefficient (Lecture 7)
        device:        torch.device or None (auto-detect)
        step_size:     StepLR epoch interval for LR decay
        gamma:         StepLR multiplicative decay factor
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader,
        val_loader,
        lr:           float = 1e-3,
        weight_decay: float = 1e-4,
        device=None,
        step_size:    int   = 10,
        gamma:        float = 0.50,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Training on: {self.device}")

        self.model        = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader

        # Loss: Cross-Entropy (Lecture 4 — softmax + negative log-likelihood)
        self.criterion = nn.CrossEntropyLoss()

        # Optimiser: Adam (Lecture 6 — combines momentum + adaptive LR)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # LR schedule: step decay (Lecture 6)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma
        )

        self.history    = defaultdict(list)
        self.best_val_acc = 0.0
        self.best_state   = None

    # ── Single epoch ──────────────────────────────────────────────────────────

    def _train_epoch(self):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in self.train_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(imgs)
            loss   = self.criterion(logits, labels)
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def _eval_epoch(self, loader=None):
        self.model.eval()
        loader = loader or self.val_loader
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits = self.model(imgs)
            loss   = self.criterion(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            preds       = logits.argmax(1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        return (
            total_loss / total,
            correct / total,
            np.array(all_preds),
            np.array(all_labels),
        )

    # ── Full training run ─────────────────────────────────────────────────────

    def fit(
        self,
        n_epochs:  int           = 40,
        patience:  int           = 10,
        save_path: Optional[str] = None,
    ) -> dict:
        """
        Train for up to n_epochs with early stopping.

        Args:
            n_epochs:  maximum training epochs
            patience:  stop if val_acc does not improve for this many epochs
            save_path: file path to save the best model checkpoint (.pt)

        Returns:
            history dict with keys: train_loss, train_acc, val_loss, val_acc, lr
        """
        no_improve = 0

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._train_epoch()
            val_loss,   val_acc, _, _ = self._eval_epoch()
            self.scheduler.step()

            cur_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(cur_lr)

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{n_epochs}"
                f"  train loss {train_loss:.4f}  acc {train_acc:.4f}"
                f"  val loss {val_loss:.4f}  acc {val_acc:.4f}"
                f"  lr {cur_lr:.2e}  [{elapsed:.1f}s]"
            )

            # Checkpoint best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_state   = {k: v.cpu().clone()
                                     for k, v in self.model.state_dict().items()}
                no_improve = 0

                if save_path:
                    torch.save(
                        {
                            "epoch":             epoch,
                            "model_state_dict":  self.best_state,
                            "val_acc":           val_acc,
                            "history":           dict(self.history),
                        },
                        save_path,
                    )
                    print(f"  ✓ Best model saved  (val_acc={val_acc:.4f})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}  "
                          f"(best val_acc={self.best_val_acc:.4f})")
                    break

        # Restore best weights
        if self.best_state:
            self.model.load_state_dict(self.best_state)

        return dict(self.history)

    # ── Evaluation helpers ────────────────────────────────────────────────────

    def evaluate_full(self, loader=None, class_names=None) -> dict:
        """
        Return loss, accuracy, per-class report, and confusion matrix.

        Args:
            loader:      DataLoader to evaluate (defaults to val_loader)
            class_names: list of string labels for report

        Returns:
            dict with keys: loss, accuracy, report, confusion_matrix
        """
        loss, acc, preds, labels = self._eval_epoch(loader)
        report = classification_report(
            labels, preds,
            target_names=class_names,
            zero_division=0,
        )
        cm = confusion_matrix(labels, preds)
        return {
            "loss":             loss,
            "accuracy":         acc,
            "report":           report,
            "confusion_matrix": cm,
            "preds":            preds,
            "labels":           labels,
        }
