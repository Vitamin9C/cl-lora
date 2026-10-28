"""Saved datasets: run experiments on a machine without the BigEarthNet LMDB.

``save_country_datasets`` samples the patches of each country once, applies the
model's evaluation transform, and stores the result as ``<dir>/<country>/train_data.pt``
and ``<dir>/<country>/test_data.pt`` (each a tuple of images and labels).
``load_saved_datasets`` reads them back with the same interface as ``get_datasets``.

The saved images are not augmented: random augmentations are lost when data is saved.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .datasets import load_country_train_val


class TensorPairDataset(Dataset):
    """Dataset over a saved (images, labels) tuple."""

    def __init__(self, data: tuple[torch.Tensor, torch.Tensor]):
        self.images, self.labels = data[0], data[1]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


def save_country_datasets(
    countries: list[str],
    save_dir: str | Path,
    train_samples: int,
    test_samples: int,
    seed: int = 42,
    model_type: Literal["spectral_gpt", "softcon"] = "spectral_gpt",
) -> None:
    """Sample, transform and save the train and test data of each country."""
    save_dir = Path(os.path.expanduser(str(save_dir)))
    for country in countries:
        out = save_dir / country
        out.mkdir(parents=True, exist_ok=True)
        for split, n in (("train", train_samples), ("test", test_samples)):
            # train_frac=1.0 keeps every sampled patch in the first dataset
            ds, _ = load_country_train_val(
                country,
                n,
                seed=seed,
                split=split,
                train_frac=1.0,
                model_type=model_type,
                augment=False,
            )
            images, labels = next(iter(DataLoader(ds, batch_size=len(ds), shuffle=False)))
            torch.save((images, labels), out / f"{split}_data.pt")


def load_saved_datasets(
    countries: list[str],
    permutation: list[int],
    saved_dir: str | Path,
    split: Literal["train", "test"] = "train",
    train_frac: float = 0.9,
    seed: int = 0,
) -> tuple[list[Dataset], list[Dataset]]:
    """Load saved datasets in permutation order.

    For ``split="train"`` the saved training data is split into train and validation
    parts by ``train_frac``; for ``split="test"`` the second list holds empty datasets.
    """
    saved_dir = Path(os.path.expanduser(str(saved_dir)))
    first, second = [], []
    for idx in permutation:
        data = torch.load(saved_dir / countries[idx] / f"{split}_data.pt")
        ds = TensorPairDataset(data)
        g = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(ds), generator=g).tolist()
        cut = int(len(ds) * train_frac)
        first.append(Subset(ds, order[:cut]))
        second.append(Subset(ds, order[cut:]))
    return first, second
