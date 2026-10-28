"""Dataset loading utilities for CL-LoRA."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from configilm.extra.DataSets.BENv2_DataSet import BENv2DataSet

from .constants import get_datapath
from .transforms import get_train_transform, get_val_transform


def load_country_train_val(
    country: str,
    n_samples: int,
    seed: int | None = None,
    include_snowy: bool = False,
    include_cloudy: bool = False,
    split: Literal["train", "test"] = "train",
    train_frac: float = 0.8,
    model_type: Literal["spectral_gpt", "softcon"] = "spectral_gpt",
    augment: bool = True,
) -> tuple[BENv2DataSet, BENv2DataSet]:
    """Load country-specific train/val datasets (augment=False: no train augmentation)."""
    # Imported here so that saved datasets and merging work without configilm installed
    from configilm.extra.DataSets.BENv2_DataSet import BENv2DataSet

    datapath = get_datapath()
    meta = pd.read_parquet(datapath["metadata_parquet"])

    img_size = 224 if model_type.lower() == "softcon" else 128
    img_size_ben = (14, img_size, img_size)

    val_transform = get_val_transform(model_type, img_size)
    train_transform = get_train_transform(model_type, img_size) if augment else val_transform

    mask = (meta.country == country) & (meta.split == split)
    available = meta.loc[mask, "patch_id"].tolist()

    if not available:
        raise ValueError(f"No {split.upper()} patches found for {country}")

    if n_samples > len(available):
        n_samples = len(available)

    rng = random.Random(seed)
    sampled = rng.sample(available, k=n_samples)
    split_at = int(n_samples * train_frac)
    train_ids = set(sampled[:split_at])
    val_ids = set(sampled[split_at:])

    def _make_ds(keep_ids: set, transform) -> BENv2DataSet:
        return BENv2DataSet(
            data_dirs=datapath,
            img_size=img_size_ben,
            split=split,
            include_snowy=include_snowy,
            include_cloudy=include_cloudy,
            patch_prefilter=lambda pid: pid in keep_ids,
            transform=transform,
        )

    return _make_ds(train_ids, train_transform), _make_ds(val_ids, val_transform)


def get_datasets(
    countries: list[str],
    permutation: list[int],
    split: Literal["train", "test"] = "train",
    include_snowy: bool = False,
    include_cloudy: bool = False,
    samples_per_country: int | None = None,
    seed: int = 0,
    train_frac: float = 0.8,
    model_type: Literal["spectral_gpt", "softcon"] = "spectral_gpt",
    saved_dir: str | None = None,
) -> tuple[list[BENv2DataSet], list[BENv2DataSet]]:
    """Load datasets for multiple countries, in permutation order.

    With ``saved_dir`` the datasets are read from files written by
    ``cl_lora.data.saved.save_country_datasets`` instead of the BigEarthNet LMDB.
    """
    if saved_dir:
        from .saved import load_saved_datasets

        return load_saved_datasets(countries, permutation, saved_dir, split, train_frac, seed)

    train_datasets, val_datasets = [], []

    for idx in permutation:
        country = countries[idx]
        train_ds, val_ds = load_country_train_val(
            country=country,
            n_samples=samples_per_country or 100000,
            seed=seed,
            include_snowy=include_snowy,
            include_cloudy=include_cloudy,
            split=split,
            train_frac=train_frac,
            model_type=model_type,
        )
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)

    return train_datasets, val_datasets
