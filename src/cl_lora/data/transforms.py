"""Data transforms for CL-LoRA."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torchvision import transforms

from .constants import S2A_MEAN, S2A_STD, SPECTRAL_MEAN, SPECTRAL_STD


class NormalizeWithStats:
    """Normalize image using mean and std statistics."""

    def __init__(self, mean: list[float] | np.ndarray, std: list[float] | np.ndarray):
        self.mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(-1, 1, 1)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img_np = img.numpy().astype(np.float32)
        img_np = (img_np - self.mean) / self.std
        return torch.from_numpy(img_np).float()


class SelectChannels:
    """Select specific channels from multi-band imagery."""

    def __init__(self, channels: list[int] | None = None):
        self.channels = channels if channels is not None else list(range(2, 14))

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return img[self.channels, :, :]


class DropSARChannels:
    """Drop SAR channels (first 2) to keep only Sentinel-2 bands."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return img[2:, :, :]


class ZeroInitializeB10:
    """Insert zero-filled B10 channel at position 9."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        b10 = torch.zeros((1, img.shape[1], img.shape[2]), dtype=img.dtype)
        return torch.cat([img[:9, :, :], b10, img[9:, :, :]], dim=0)


def get_train_transform(
    model_type: Literal["spectral_gpt", "softcon"],
    img_size: int = 224,
) -> transforms.Compose:
    """Get training transform pipeline."""
    base = [transforms.Resize((img_size, img_size))]

    augmentations = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomChoice(
            [
                transforms.RandomRotation(degrees=(0, 0)),
                transforms.RandomRotation(degrees=(90, 90)),
                transforms.RandomRotation(degrees=(180, 180)),
                transforms.RandomRotation(degrees=(270, 270)),
            ]
        ),
        transforms.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.8, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
    ]

    if model_type.lower() == "softcon":
        preprocessing = [DropSARChannels(), ZeroInitializeB10()]
        normalization = [NormalizeWithStats(S2A_MEAN, S2A_STD)]
    elif model_type.lower() in ("spectral_gpt", "spectralgpt"):
        preprocessing = [SelectChannels()]
        normalization = [NormalizeWithStats(SPECTRAL_MEAN, SPECTRAL_STD)]
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return transforms.Compose(base + preprocessing + augmentations + normalization)


def get_val_transform(
    model_type: Literal["spectral_gpt", "softcon"],
    img_size: int = 224,
) -> transforms.Compose:
    """Get validation transform pipeline (no augmentations)."""
    base = [transforms.Resize((img_size, img_size))]

    if model_type.lower() == "softcon":
        preprocessing = [DropSARChannels(), ZeroInitializeB10()]
        normalization = [NormalizeWithStats(S2A_MEAN, S2A_STD)]
    elif model_type.lower() in ("spectral_gpt", "spectralgpt"):
        preprocessing = [SelectChannels()]
        normalization = [NormalizeWithStats(SPECTRAL_MEAN, SPECTRAL_STD)]
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return transforms.Compose(base + preprocessing + normalization)
