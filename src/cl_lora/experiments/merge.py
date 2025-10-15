"""Merge experiment runner for continual learning with LoRA merging.

This module implements continual learning experiments using LoRA merging
strategies (ZipLoRA, LoRAHub, LoRASoups) for multi-task learning.
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from ..data import get_datasets
from ..training import CLLightningModule, create_trainer, eval_model
from .merging import LoRAHubMerger, ZipLoRAMerger, simple_average_merge

log = logging.getLogger(__name__)


def sample_subset(dataset, n: int, seed: int) -> Subset:
    """Return a Subset of size min(n, len(dataset)).

    Args:
        dataset: Source dataset
        n: Maximum number of samples
        seed: Random seed for reproducibility

    Returns:
        Subset of the dataset
    """
    if n >= len(dataset):
        return dataset
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[:n]
    return Subset(dataset, idx.tolist())


def run_merge_experiment(
    config: dict[str, Any],
    merge_type: Literal["ZipLoRA", "LoRAHub", "LoRASoups"] = "ZipLoRA",
) -> pd.DataFrame:
    """Run continual learning experiment with LoRA merging.

    At each step after the first, we have:
    - A model trained on all previous tasks (model_old)
    - A model trained on the current task (model_new)

    We merge these using the specified strategy to create a combined model.

    Args:
        config: Configuration dictionary with experiment parameters
        merge_type: Type of merging strategy to use

    Returns:
        DataFrame with results for each step
    """
    # Extract config
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    exp_cfg = config.get("experiment", {})

    backbone = model_cfg.get("backbone", "spectral_gpt")
    lora_r = model_cfg.get("lora_r", 4)
    lora_alpha = model_cfg.get("lora_alpha", 16)
    num_classes = model_cfg.get("num_classes", 19)

    batch_size = data_cfg.get("batch_size", 64)
    num_workers = data_cfg.get("num_workers", 4)

    epochs = train_cfg.get("epochs", 15)
    lr = float(train_cfg.get("lr", 1e-3))

    countries = exp_cfg.get("countries", [])
    permutation = exp_cfg.get("permutation", list(range(len(countries))))
    train_samples = exp_cfg.get("train_samples", 10000)
    test_samples = exp_cfg.get("test_samples", 5000)
    memory_size = exp_cfg.get("memory_size", 500)
    seed = exp_cfg.get("seed", 42)

    save_dir = os.getenv("PATHS_SAVE_DIR", "./saved_models")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model function based on backbone
    if backbone.lower() == "spectral_gpt":
        from ..models.backbones import load_spectral_gpt

        load_model_fn = lambda: load_spectral_gpt(
            lora_r=lora_r, lora_alpha=lora_alpha, num_classes=num_classes, device=device
        )
    else:
        from ..models.backbones import load_softcon

        load_model_fn = lambda: load_softcon(
            lora_r=lora_r, lora_alpha=lora_alpha, num_classes=num_classes, device=device
        )

    # Load all datasets
    log.info("Loading datasets...")
    train_sets, val_sets = get_datasets(
        countries=countries,
        permutation=permutation,
        split="train",
        samples_per_country=train_samples,
        seed=seed,
        train_frac=0.9,
        model_type=backbone,
    )
    test_sets, _ = get_datasets(
        countries=countries,
        permutation=permutation,
        split="test",
        samples_per_country=test_samples,
        seed=seed,
        train_frac=1.0,
        model_type=backbone,
    )

    # Initialize merger
    if merge_type == "ZipLoRA":
        merger = ZipLoRAMerger(device=device)
    elif merge_type == "LoRAHub":
        merger = LoRAHubMerger(device=device)
    else:  # LoRASoups
        merger = None  # Use simple_average_merge

    results = []
    trained_models = []  # Store models for each task

    # Step 1: Train on first task
    log.info(f"Step 1: Training on {countries[permutation[0]]}")
    model = load_model_fn()
    pl_model = CLLightningModule(model, num_classes=num_classes, lr=lr)

    train_loader = DataLoader(
        train_sets[0], batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_sets[0], batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    trainer = create_trainer(max_epochs=epochs, save_dir=save_dir)
    trainer.fit(pl_model, train_loader, val_loader)
    trained_models.append(pl_model)

    # Evaluate step 1
    test_loader = DataLoader(
        test_sets[0], batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    metrics = eval_model(pl_model, test_loader, device)
    results.append(
        {
            "step": 1,
            "countries": (countries[permutation[0]],),
            f"{countries[permutation[0]]}_micro_ap": metrics["micro_ap"],
            "mean_micro_ap": metrics["micro_ap"],
        }
    )
    log.info(f"Step 1 result: {results[-1]}")

    # Steps 2+: Train new task and merge
    for step in range(2, len(permutation) + 1):
        seen_idx = permutation[:step]
        seen_countries = [countries[i] for i in seen_idx]
        current_country = countries[permutation[step - 1]]
        log.info(f"Step {step}: Training on {current_country}, merging with previous")

        # Train model on current task
        model_new = load_model_fn()
        pl_model_new = CLLightningModule(model_new, num_classes=num_classes, lr=lr)

        train_loader_new = DataLoader(
            train_sets[step - 1], batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader_new = DataLoader(
            val_sets[step - 1], batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        trainer = create_trainer(max_epochs=epochs, save_dir=save_dir)
        trainer.fit(pl_model_new, train_loader_new, val_loader_new)

        # Prepare replay data from old tasks
        share_per_task = math.ceil(memory_size / (step - 1))
        replay_sets = [sample_subset(train_sets[t], share_per_task, seed) for t in range(step - 1)]
        concat_replay = ConcatDataset(replay_sets)
        replay_loader = DataLoader(
            concat_replay, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        # Merge models
        start_merge = time.time()
        if merge_type == "ZipLoRA":
            merged_model = merger.merge(
                trained_models[-1].model,
                pl_model_new.model,
                replay_loader,
                train_loader_new,
            )
            pl_merged = CLLightningModule(merged_model, num_classes=num_classes, lr=lr)
        elif merge_type == "LoRAHub":
            all_models = [m.model for m in trained_models] + [pl_model_new.model]
            val_concat = ConcatDataset([val_sets[t] for t in range(step)])
            val_loader_all = DataLoader(
                val_concat, batch_size=batch_size, shuffle=True, num_workers=num_workers
            )
            merged_model, coefficients = merger.merge(
                load_model_fn(),  # Fresh base model
                all_models,
                val_loader_all,
            )
            pl_merged = CLLightningModule(merged_model, num_classes=num_classes, lr=lr)
            log.info(f"LoRAHub coefficients: {coefficients}")
        else:  # LoRASoups
            all_models = [m.model for m in trained_models] + [pl_model_new.model]
            merged_model = simple_average_merge(all_models)
            pl_merged = CLLightningModule(merged_model, num_classes=num_classes, lr=lr)

        merge_time = time.time() - start_merge
        log.info(f"Merge time: {merge_time:.2f}s")

        trained_models.append(pl_merged)

        # Save merged model
        save_path = Path(save_dir) / f"merged_step{step}_{merge_type}.pkl"
        joblib.dump(pl_merged.model.state_dict(), save_path)
        log.info(f"Saved: {save_path}")

        # Evaluate on all seen countries
        eval_metrics = {}
        for idx in seen_idx:
            country = countries[idx]
            test_loader = DataLoader(
                test_sets[idx], batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
            m = eval_model(pl_merged, test_loader, device)
            eval_metrics[f"{country}_micro_ap"] = m["micro_ap"]

        mean_ap = sum(v for k, v in eval_metrics.items() if "micro_ap" in k) / step

        row = {
            "step": step,
            "countries": tuple(seen_countries),
            "merge_time_s": merge_time,
            "mean_micro_ap": mean_ap,
        }
        row.update(eval_metrics)
        results.append(row)
        log.info(f"Result: {row}")

    return pd.DataFrame(results)


def main():
    """CLI entry point for merge experiments."""
    import argparse

    import yaml
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Run LoRA merge experiment")
    parser.add_argument("--config", "-c", required=True, help="Path to config file")
    parser.add_argument(
        "--merge-type",
        "-m",
        choices=["ZipLoRA", "LoRAHub", "LoRASoups"],
        default="ZipLoRA",
        help="Merge strategy to use",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Handle inheritance
    if "_base_" in config:
        base_path = Path(args.config).parent / config.pop("_base_")
        with open(base_path) as f:
            base_config = yaml.safe_load(f)
        for key, value in config.items():
            if isinstance(value, dict) and key in base_config:
                base_config[key].update(value)
            else:
                base_config[key] = value
        config = base_config

    log.info(f"Running {args.merge_type} merge experiment")
    log.info(f"Config: {config}")

    results_df = run_merge_experiment(config, args.merge_type)
    print(results_df.to_string())

    # Save results
    save_dir = os.getenv("PATHS_SAVE_DIR", "./saved_models")
    results_path = Path(save_dir) / f"results_{args.merge_type}.csv"
    results_df.to_csv(results_path, index=False)
    log.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
