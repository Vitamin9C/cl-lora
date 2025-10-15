"""Trainer factory for creating configured PyTorch Lightning trainers.

Provides a centralized trainer creation with consistent callbacks for
early stopping, checkpointing, and learning rate monitoring.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as L
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)


def create_trainer(
    max_epochs: int = 15,
    save_dir: str = "./saved_models",
    patience: int = 5,
    min_delta: float = 0.01,
    log_every_n_steps: int = 50,
    accelerator: str = "auto",
    strategy: str | None = None,
    **kwargs: Any,
) -> L.Trainer:
    """Create and configure a PyTorch Lightning Trainer.

    Sets up a trainer with:
    - Early stopping based on validation loss
    - Model checkpointing (saves best model)
    - Learning rate monitoring
    - DDP strategy for multi-GPU training

    Args:
        max_epochs: Maximum number of training epochs
        save_dir: Directory for saving checkpoints and logs
        patience: Early stopping patience (epochs without improvement)
        min_delta: Minimum change to qualify as improvement
        log_every_n_steps: Logging frequency
        accelerator: Accelerator type ("auto", "gpu", "cpu")
        strategy: Training strategy (None for auto, "ddp" for distributed)
        **kwargs: Additional arguments passed to L.Trainer

    Returns:
        Configured Lightning Trainer instance.

    Example:
        >>> trainer = create_trainer(max_epochs=25, save_dir="./outputs")
        >>> trainer.fit(model, train_loader, val_loader)
    """
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        min_delta=min_delta,
        check_on_train_epoch_end=True,
        verbose=True,
    )

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=save_dir,
        filename="{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
        verbose=True,
    )

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Determine strategy
    if strategy is None:
        # Use DDP if multiple GPUs available
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            strategy = L.strategies.DDPStrategy(find_unused_parameters=True)
        else:
            strategy = "auto"
    elif strategy == "ddp":
        strategy = L.strategies.DDPStrategy(find_unused_parameters=True)

    # Create trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        strategy=strategy,
        log_every_n_steps=log_every_n_steps,
        default_root_dir=save_dir,
        callbacks=[early_stopping, checkpoint_callback, lr_monitor],
        **kwargs,
    )

    return trainer


def create_trainer_from_config(config: dict[str, Any]) -> L.Trainer:
    """Create trainer from configuration dictionary.

    Args:
        config: Configuration dictionary with training parameters.
               Expected keys: epoch, save_dir, batch_size, etc.

    Returns:
        Configured Lightning Trainer instance.
    """
    return create_trainer(
        max_epochs=config.get("epoch", 15),
        save_dir=config.get("save_dir", "./saved_models"),
        patience=config.get("patience", 5),
        log_every_n_steps=config.get("log_every_n_steps", 50),
    )
