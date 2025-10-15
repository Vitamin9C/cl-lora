"""Constants for CL-LoRA: band information, statistics, and data paths."""

from __future__ import annotations

import os
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()

# Sentinel-2 Band Names
ALL_BANDS_S2_L2A: list[str] = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
]

# SoftCon Normalization Statistics (13 bands including B10)
S2A_MEAN: list[float] = [
    752.40087073,
    884.29673756,
    1144.16202635,
    1297.47289228,
    1624.90992062,
    2194.6423161,
    2422.21248945,
    2517.76053101,
    2581.64687018,
    2645.51888987,
    0,
    2368.51236873,
    1805.06846033,
]

S2A_STD: list[float] = [
    1108.02887453,
    1155.15170768,
    1183.6292542,
    1368.11351514,
    1370.265037,
    1355.55390699,
    1416.51487101,
    1474.78900051,
    1439.3086061,
    1582.28010962,
    1,
    1455.52084939,
    1343.48379601,
]

# SpectralGPT Normalization Statistics (12 bands, no B10)
SPECTRAL_MEAN: list[float] = [
    1370.19151926,
    1184.3824625,
    1120.77120066,
    1136.26026392,
    1263.73947144,
    1645.40315151,
    1846.87040806,
    1762.59530783,
    1972.62420416,
    582.72633433,
    1732.16362238,
    1247.91870117,
]

SPECTRAL_STD: list[float] = [
    633.15169573,
    650.2842772,
    712.12507725,
    965.23119807,
    948.9819932,
    1108.06650639,
    1258.36394548,
    1233.1492281,
    1364.38688993,
    472.37967789,
    1310.36996126,
    1087.6020813,
]


class DataPathDict(TypedDict):
    images_lmdb: str
    metadata_parquet: str
    metadata_snow_cloud_parquet: str


def get_datapath() -> DataPathDict:
    """Get data paths from environment variables."""
    images_lmdb = os.getenv("PATHS_IMAGES_LMDB", "")
    metadata_parquet = os.getenv("PATHS_METADATA", "")
    metadata_snow_cloud = os.getenv("PATHS_METADATA_SNOW_CLOUD", "")

    if not images_lmdb or not metadata_parquet:
        raise ValueError(
            "Required environment variables not set. "
            "Copy .env.example to .env and configure your paths."
        )

    return {
        "images_lmdb": images_lmdb,
        "metadata_parquet": metadata_parquet,
        "metadata_snow_cloud_parquet": metadata_snow_cloud,
    }


def get_pretrained_path(model_type: str) -> str:
    """Get pretrained model path from environment."""
    if model_type.lower() in ("spectral_gpt", "spectralgpt"):
        path = os.getenv("PATHS_SPECTRAL_PRETRAINED", "")
    elif model_type.lower() == "softcon":
        path = os.getenv("PATHS_SOFTCON_PRETRAINED", "")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if not path:
        raise ValueError(f"Pretrained path not set for {model_type}")
    return path


def get_save_dir() -> str:
    return os.getenv("PATHS_SAVE_DIR", "./saved_models")
