"""Baseline experiment runner for continual learning.

Implements two baseline strategies:
- Replay: Retrain from scratch with all previous data at each step
- No Replay: Sequential fine-tuning without replaying old data
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from ..data import get_datasets
from ..training import CLLightningModule, create_trainer, eval_model

log = logging.getLogger(__name__)


def run_baseline_experiment(
    config: dict[str, Any],
    experiment_type: Literal["replay", "no_replay"] = "no_replay",
) -> pd.DataFrame:
    """Run baseline continual learning experiment.

    Args:
        config: Configuration dictionary
        experiment_type: Either 'replay' (full replay) or 'no_replay' (sequential)

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
    patience = train_cfg.get("patience", 5)

    countries = exp_cfg.get("countries", [])
    permutation = exp_cfg.get("permutation", list(range(len(countries))))
    train_samples = exp_cfg.get("train_samples", 10000)
    test_samples = exp_cfg.get("test_samples", 5000)
    seed = exp_cfg.get("seed", 42)

    save_dir = os.getenv("PATHS_SAVE_DIR", "./saved_models")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model function
    if backbone.lower() in ("spectral_gpt", "spectralgpt"):
        from ..models.backbones import load_spectral_gpt

        load_model_fn = lambda: load_spectral_gpt(
            lora_r=lora_r, lora_alpha=lora_alpha, num_classes=num_classes, device=device
        )
    else:
        from ..models.backbones import load_softcon

        load_model_fn = lambda: load_softcon(
            lora_r=lora_r, lora_alpha=lora_alpha, num_classes=num_classes, device=device
        )

    # Load datasets
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

    results = []
    current_model = None

    for step in range(1, len(permutation) + 1):
        seen_idx = permutation[:step]
        seen_countries = [countries[i] for i in seen_idx]
        log.info(f"Step {step}: Training on {seen_countries}")

        # Prepare training data
        if experiment_type == "replay":
            # Full replay: use all data from all seen tasks
            model = load_model_fn()  # Always start fresh
            concat_train = ConcatDataset([train_sets[t] for t in range(step)])
            concat_val = ConcatDataset([val_sets[t] for t in range(step)])
        else:
            # No replay: only current task, continue from previous model
            if step == 1:
                model = load_model_fn()
            else:
                model = current_model.model  # Continue training
            concat_train = train_sets[step - 1]
            concat_val = val_sets[step - 1]

        train_loader = DataLoader(
            concat_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            concat_val, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Train
        pl_model = CLLightningModule(model, num_classes=num_classes, lr=lr)
        trainer = create_trainer(max_epochs=epochs, save_dir=save_dir, patience=patience)

        start_train = time.time()
        trainer.fit(pl_model, train_loader, val_loader)
        train_time = time.time() - start_train

        current_model = pl_model

        # Save model
        save_path = Path(save_dir) / f"{experiment_type}_step{step}.pkl"
        joblib.dump(pl_model.model.state_dict(), save_path)

        # Evaluate on all seen tasks
        eval_metrics = {}
        for idx in seen_idx:
            country = countries[idx]
            test_loader = DataLoader(
                test_sets[idx], batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
            m = eval_model(pl_model, test_loader, device)
            eval_metrics[f"{country}_micro_ap"] = m["micro_ap"]
            eval_metrics[f"{country}_accuracy"] = m["accuracy"]

        mean_ap = sum(v for k, v in eval_metrics.items() if "micro_ap" in k) / step

        row = {
            "step": step,
            "countries": tuple(seen_countries),
            "train_time_s": train_time,
            "mean_micro_ap": mean_ap,
        }
        row.update(eval_metrics)
        results.append(row)
        log.info(f"Result: {row}")

    return pd.DataFrame(results)


def main():
    """CLI entry point for baseline experiments."""
    import argparse

    import yaml
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Run baseline CL experiment")
    parser.add_argument("--config", "-c", required=True, help="Path to config file")
    parser.add_argument(
        "--type", "-t", choices=["replay", "no_replay"], default="no_replay", help="Experiment type"
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

    log.info(f"Running {args.type} experiment")

    results_df = run_baseline_experiment(config, args.type)
    print(results_df.to_string())

    # Save results
    save_dir = os.getenv("PATHS_SAVE_DIR", "./saved_models")
    results_path = Path(save_dir) / f"results_{args.type}.csv"
    results_df.to_csv(results_path, index=False)
    log.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
