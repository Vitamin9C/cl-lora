"""Baseline experiment runner for continual learning.

Implements three strategies:
- Replay: Retrain from scratch with all previous data at each step (upper bound)
- No Replay: Sequential fine-tuning without replaying old data (lower bound)
- Task tuning: Train one independent LoRA per country; its heads are the input of
  the merge experiments
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
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from ..data import get_datasets
from ..models.lora_heads import save_head
from ..training import CLLightningModule, create_trainer, eval_model

log = logging.getLogger(__name__)

EXPERIMENT_TYPES = ("replay", "no_replay", "task_tuning")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config; a ``_base_`` key names a config whose sections it updates."""
    path = Path(path)
    with open(path) as f:
        config = yaml.safe_load(f)
    if "_base_" in config:
        base = load_config(path.parent / config.pop("_base_"))
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key].update(value)
            else:
                base[key] = value
        config = base
    return config


def run_baseline_experiment(
    config: dict[str, Any],
    experiment_type: Literal["replay", "no_replay", "task_tuning"] = "no_replay",
) -> pd.DataFrame:
    """Run a baseline continual learning experiment.

    Args:
        config: Configuration dictionary
        experiment_type: 'replay' (full replay), 'no_replay' (sequential) or
            'task_tuning' (one independent model per country)

    Returns:
        DataFrame with results for each step
    """
    if experiment_type not in EXPERIMENT_TYPES:
        raise ValueError(
            f"Unknown experiment type {experiment_type}; use one of {EXPERIMENT_TYPES}"
        )
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    exp_cfg = config.get("experiment", {})

    backbone = model_cfg.get("backbone", "spectral_gpt")
    lora_r = model_cfg.get("lora_r", 4)
    lora_alpha = model_cfg.get("lora_alpha", 16)
    num_classes = model_cfg.get("num_classes", 19)

    batch_size = data_cfg.get("batch_size", 16)
    num_workers = data_cfg.get("num_workers", 4)
    saved_dir = data_cfg.get("saved_datasets_dir") or None

    epochs = train_cfg.get("epochs", 15)
    lr = float(train_cfg.get("lr", 1e-4))
    patience = train_cfg.get("patience")  # None: no early stopping

    countries = exp_cfg.get("countries", [])
    permutation = exp_cfg.get("permutation", list(range(len(countries))))
    seed = exp_cfg.get("seed", 42)

    permutation_str = "_".join(map(str, permutation))
    save_dir = (
        Path(os.getenv("PATHS_SAVE_DIR", "./saved_models"))
        / experiment_type
        / f"permutation_{permutation_str}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if backbone.lower() in ("spectral_gpt", "spectralgpt"):
        from ..models.backbones import load_spectral_gpt as loader
    else:
        from ..models.backbones import load_softcon as loader

    def load_model_fn():
        return loader(lora_r=lora_r, lora_alpha=lora_alpha, num_classes=num_classes, device=device)

    # Datasets are returned in permutation order: index t is the t-th country of the run.
    common = dict(
        countries=countries,
        permutation=permutation,
        seed=seed,
        model_type=backbone,
        include_snowy=data_cfg.get("include_snowy", False),
        include_cloudy=data_cfg.get("include_cloudy", False),
        saved_dir=saved_dir,
    )
    log.info("Loading datasets...")
    train_sets, val_sets = get_datasets(
        split="train",
        samples_per_country=exp_cfg.get("train_samples"),
        train_frac=exp_cfg.get("train_frac", 0.9),
        **common,
    )
    test_sets, _ = get_datasets(
        split="test", samples_per_country=exp_cfg.get("test_samples"), train_frac=1.0, **common
    )
    ordered = [countries[i] for i in permutation]

    results = []
    current_model = None

    for step in range(1, len(ordered) + 1):
        if experiment_type == "task_tuning":
            eval_countries = [ordered[step - 1]]
            eval_sets = [test_sets[step - 1]]
            log.info(f"Step {step}: Training independently on {ordered[step - 1]}")
        else:
            eval_countries = ordered[:step]
            eval_sets = test_sets[:step]
            log.info(f"Step {step}: Training on {eval_countries}")

        # Prepare training data
        if experiment_type == "replay":
            # Full replay: use all data from all seen tasks
            model = load_model_fn()  # Always start fresh
            concat_train = ConcatDataset(train_sets[:step])
            concat_val = ConcatDataset(val_sets[:step])
        elif experiment_type == "task_tuning":
            # Independent model per country
            model = load_model_fn()
            concat_train = train_sets[step - 1]
            concat_val = val_sets[step - 1]
        else:
            # No replay: only current task, continue from previous model
            model = load_model_fn() if step == 1 else current_model.model  # Continue training
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
        trainer = create_trainer(max_epochs=epochs, save_dir=str(save_dir), patience=patience)

        start_train = time.time()
        trainer.fit(pl_model, train_loader, val_loader)
        train_time = time.time() - start_train

        current_model = pl_model

        # Save model and its head (LoRA + classifier)
        if trainer.is_global_zero:
            if experiment_type == "task_tuning":
                prefix = save_dir / ordered[step - 1]
            else:
                prefix = save_dir / f"step{step}"
            save_head(pl_model.model, f"{prefix}_lora.safetensors", f"{prefix}_fc.safetensors")
            joblib.dump(pl_model.model.state_dict(), save_dir / f"{experiment_type}_step{step}.pkl")
            log.info(f"Saved head: {prefix}_lora.safetensors / {prefix}_fc.safetensors")

        # Evaluate on the seen tasks (task tuning: only the current task)
        eval_metrics = {}
        for country, test_set in zip(eval_countries, eval_sets):
            test_loader = DataLoader(
                test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
            m = eval_model(pl_model, test_loader, device)
            eval_metrics[f"{country}_micro_ap"] = m["micro_ap"]
            eval_metrics[f"{country}_macro_ap"] = m["macro_ap"]
            eval_metrics[f"{country}_accuracy"] = m["accuracy"]

        n = len(eval_countries)
        row = {
            "step": step,
            "countries": tuple(eval_countries),
            "train_time_s": train_time,
            "mean_micro_ap": sum(v for k, v in eval_metrics.items() if k.endswith("_micro_ap")) / n,
        }
        row.update(eval_metrics)
        results.append(row)
        log.info(f"Result: {row}")

    return pd.DataFrame(results)


def main():
    """CLI entry point for baseline experiments."""
    import argparse

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Run baseline CL experiment")
    parser.add_argument("--config", "-c", required=True, help="Path to config file")
    parser.add_argument(
        "--type",
        "-t",
        choices=EXPERIMENT_TYPES,
        default=None,
        help="Experiment type (default: experiment.type in the config)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_config(args.config)
    experiment_type = args.type or config.get("experiment", {}).get("type", "no_replay")

    log.info(f"Running {experiment_type} experiment")

    results_df = run_baseline_experiment(config, experiment_type)
    print(results_df.to_string())

    # Save results
    save_dir = os.getenv("PATHS_SAVE_DIR", "./saved_models")
    results_path = Path(save_dir) / f"results_{experiment_type}.csv"
    results_df.to_csv(results_path, index=False)
    log.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
