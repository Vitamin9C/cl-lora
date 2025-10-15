"""Unified Lightning module for continual learning experiments.

This module provides a single LightningModule that works with both
SpectralGPT and SoftCon backbones for multi-label classification.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as L
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score


class CLLightningModule(L.LightningModule):
    """Lightning module for continual learning with LoRA-adapted Vision Transformers.

    Supports both SpectralGPT and SoftCon backbones with unified training interface.
    Uses BCEWithLogitsLoss for multi-label classification.

    Args:
        model: LoRA-wrapped model (SpectralGPT or SoftCon)
        num_classes: Number of output classes (default: 19 for BigEarthNet)
        lr: Learning rate (default: 1e-4)
        use_scheduler: Whether to use ReduceLROnPlateau scheduler

    Example:
        >>> model = load_spectral_gpt(num_classes=19)
        >>> pl_model = CLLightningModule(model, num_classes=19, lr=1e-4)
        >>> trainer.fit(pl_model, train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 19,
        lr: float = 1e-4,
        use_scheduler: bool = True,
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.use_scheduler = use_scheduler
        self.criterion = nn.BCEWithLogitsLoss()

        # Save hyperparameters (excluding model to avoid serialization issues)
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Logits tensor of shape [B, num_classes]
        """
        return self.model(x)

    def _compute_metrics(self, outputs: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        """Compute metrics for a batch.

        Args:
            outputs: Model logits [B, num_classes]
            labels: Ground truth labels [B, num_classes]

        Returns:
            Dictionary with loss, accuracy, and micro AP
        """
        loss = self.criterion(outputs, labels.float())

        # Predictions and probabilities
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        # Accuracy (per-label accuracy)
        acc = (preds == labels).float().mean()

        # Micro AP (computed on CPU for sklearn)
        try:
            micro_ap = average_precision_score(
                labels.cpu().numpy(), probs.detach().cpu().numpy(), average="micro"
            )
        except Exception:
            micro_ap = 0.0

        return {
            "loss": loss,
            "accuracy": acc,
            "micro_ap": micro_ap,
        }

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Tuple of (images, labels)
            batch_idx: Batch index

        Returns:
            Loss tensor for backpropagation
        """
        imgs, labels = batch
        outputs = self(imgs)
        metrics = self._compute_metrics(outputs, labels)

        # Log metrics
        self.log("train_loss", metrics["loss"], prog_bar=True, sync_dist=True)
        self.log("train_acc", metrics["accuracy"], prog_bar=True, sync_dist=True)
        self.log("train_micro_ap", metrics["micro_ap"], prog_bar=True, sync_dist=True)

        return metrics["loss"]

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Validation step.

        Args:
            batch: Tuple of (images, labels)
            batch_idx: Batch index
        """
        imgs, labels = batch
        outputs = self(imgs)
        metrics = self._compute_metrics(outputs, labels)

        # Log metrics
        self.log("val_loss", metrics["loss"], prog_bar=True, sync_dist=True)
        self.log("val_acc", metrics["accuracy"], prog_bar=True, sync_dist=True)
        self.log("val_micro_ap", metrics["micro_ap"], prog_bar=True, sync_dist=True)

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """Test step.

        Args:
            batch: Tuple of (images, labels)
            batch_idx: Batch index
        """
        imgs, labels = batch
        outputs = self(imgs)
        metrics = self._compute_metrics(outputs, labels)

        # Log metrics
        self.log("test_loss", metrics["loss"], sync_dist=True)
        self.log("test_acc", metrics["accuracy"], sync_dist=True)
        self.log("test_micro_ap", metrics["micro_ap"], sync_dist=True)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary with optimizer and optionally lr_scheduler configuration.
        """
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )

        if not self.use_scheduler:
            return {"optimizer": optimizer}

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=2,
            threshold=0.01,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


# Backward compatibility aliases
SpectralGPTLightningModule = CLLightningModule
SoftConLightningModule = CLLightningModule
