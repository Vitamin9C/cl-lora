#!/usr/bin/env python
"""Save sampled, transformed datasets so experiments can run without the LMDB.

Usage:
    python scripts/save_datasets.py --config configs/default.yaml --out ~/saved_datasets

Then set data.saved_datasets_dir in the experiment config to the same directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

from cl_lora.data import save_country_datasets
from cl_lora.experiments.baseline import load_config

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Save datasets for another machine")
    parser.add_argument("--config", "-c", default="configs/default.yaml")
    parser.add_argument("--out", "-o", required=True, help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp = cfg["experiment"]
    save_country_datasets(
        countries=exp["countries"],
        save_dir=args.out,
        train_samples=exp["train_samples"],
        test_samples=exp["test_samples"],
        seed=exp.get("seed", 42),
        model_type=cfg["model"].get("backbone", "spectral_gpt"),
    )
    print(f"Saved datasets to {args.out}")
