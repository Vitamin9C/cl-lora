"""Merge experiment runner for continual learning with LoRA merging.

Inputs are the heads of a task-tuning run (``experiment.type: task_tuning`` in the
baseline runner): one LoRA and one classifier per country, saved as
``<country>_lora.safetensors`` and ``<country>_fc.safetensors``.

Two approaches:

- continual: at step t the merge result of steps 1..t-1 is merged with the head of
  country t, using a replay memory of old data (``memory_size`` samples in total,
  split equally over the old countries) and the training data of country t.
- from_scratch: at step t the heads of countries 1..t are merged at once, using the
  pooled training data of these countries.

After each step the merged model is evaluated on the test data of every seen country.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from ..data import get_datasets
from ..models.lora_heads import load_head
from ..training import eval_model
from .merging import MERGE_TYPES, merge_heads

log = logging.getLogger(__name__)


def sample_subset(dataset, n: int, seed: int):
    """Return a random subset of size min(n, len(dataset))."""
    if n >= len(dataset):
        return dataset
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[:n]
    return Subset(dataset, idx.tolist())


def evaluate_seen(model, test_sets, seen_countries, batch_size, num_workers, device) -> dict:
    """Evaluate on the test set of every seen country, on all of them pooled, and the mean."""
    metrics: dict[str, float] = {}
    pooled = DataLoader(ConcatDataset(test_sets), batch_size=batch_size, num_workers=num_workers)
    for name, val in eval_model(model, pooled, device).items():
        metrics[f"combined_{name}"] = val
    sums, counts = defaultdict(float), defaultdict(int)
    for country, test_set in zip(seen_countries, test_sets):
        loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
        for name, val in eval_model(model, loader, device).items():
            metrics[f"{country}_{name}"] = val
            sums[name] += val
            counts[name] += 1
    metrics.update({f"mean_{k}": sums[k] / counts[k] for k in sums})
    return metrics


def run_merge_experiment(
    config: dict[str, Any],
    merge_type: Literal["ZipLoRA", "LoRAHub", "LoRASoups"] = "ZipLoRA",
    approach: Literal["continual", "from_scratch"] | None = None,
) -> pd.DataFrame:
    """Run a LoRA merging experiment and return one result row per step."""
    if merge_type not in MERGE_TYPES:
        raise ValueError(f"Unknown merge type {merge_type}; use one of {MERGE_TYPES}")
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    exp_cfg = config.get("experiment", {})
    merge_cfg = config.get("merge", {})

    backbone = model_cfg.get("backbone", "spectral_gpt")
    lora_r = model_cfg.get("lora_r", 4)
    lora_alpha = model_cfg.get("lora_alpha", 16)
    num_classes = model_cfg.get("num_classes", 19)
    batch_size = data_cfg.get("batch_size", 16)
    num_workers = data_cfg.get("num_workers", 4)
    saved_dir = data_cfg.get("saved_datasets_dir") or None
    epochs = merge_cfg.get("epochs", train_cfg.get("epochs", 15))
    lr = float(train_cfg.get("lr", 1e-4))
    countries = exp_cfg.get("countries", [])
    permutation = exp_cfg.get("permutation", list(range(len(countries))))
    memory_size = exp_cfg.get("memory_size", 500)
    seed = exp_cfg.get("seed", 42)
    approach = approach or merge_cfg.get("approach", "continual")
    heads_dir = Path(
        os.path.expandvars(
            os.path.expanduser(
                merge_cfg.get("task_heads_dir")
                or os.getenv("PATHS_TASK_HEADS", "./saved_models/task_tuning")
            )
        )
    )
    save_dir = Path(os.getenv("PATHS_SAVE_DIR", "./saved_models")) / f"{merge_type}_{approach}"
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
    train_sets, _ = get_datasets(
        split="train",
        samples_per_country=exp_cfg.get("train_samples"),
        train_frac=exp_cfg.get("train_frac", 0.9),
        **common,
    )
    test_sets, _ = get_datasets(
        split="test", samples_per_country=exp_cfg.get("test_samples"), train_frac=1.0, **common
    )
    ordered = [countries[i] for i in permutation]
    heads = [
        load_head(heads_dir / f"{c}_lora.safetensors", heads_dir / f"{c}_fc.safetensors")
        for c in ordered
    ]
    log.info(f"Loaded {len(heads)} task heads from {heads_dir}")

    merge_kwargs = dict(
        epochs=epochs,
        lr=lr,
        device=device,
        zip_cos_weight=float(merge_cfg.get("zip_cos_weight", 1.0)),
        zip_init=float(merge_cfg.get("zip_init", 0.5)),
        hub_budget=int(merge_cfg.get("hub_budget", 40)),
        hub_l1=float(merge_cfg.get("hub_l1", 0.05)),
    )

    def loader_of(ds, shuffle=True):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    results = []
    merged = heads[0]
    for step in range(2, len(ordered) + 1):
        seen = ordered[:step]
        log.info(f"Step {step} ({approach}): merging {seen}")
        if approach == "continual":
            share = memory_size if step == 2 else math.ceil(memory_size / (step - 1))
            replay = ConcatDataset(
                [sample_subset(train_sets[t], share, seed) for t in range(step - 1)]
            )
            new = train_sets[step - 1]
            loaders = [loader_of(replay), loader_of(new)]
            hub_data = ConcatDataset([replay, sample_subset(new, len(replay), seed)])
            to_merge, fc_weights = [merged, heads[step - 1]], [(step - 1) / step, 1 / step]
        elif approach == "from_scratch":
            pooled = ConcatDataset(train_sets[:step])
            loaders = [loader_of(pooled)]
            hub_data = sample_subset(pooled, memory_size * 2, seed)
            to_merge, fc_weights = heads[:step], [1 / step] * step
        else:
            raise ValueError(f"Unknown approach {approach}; use 'continual' or 'from_scratch'")

        start = time.time()
        merged, model = merge_heads(
            merge_type,
            load_model_fn(),
            to_merge,
            fc_weights,
            loaders,
            hub_loader=loader_of(hub_data, shuffle=False),
            **merge_kwargs,
        )
        merge_time = time.time() - start
        torch.save(merged, save_dir / f"step{step}_head.pt")

        metrics = evaluate_seen(model, test_sets[:step], seen, batch_size, num_workers, device)
        row = {"step": step, "countries": tuple(seen), "merge_time_s": merge_time}
        row.update(metrics)
        results.append(row)
        log.info(f"Result: {row}")

    return pd.DataFrame(results)


def main():
    """CLI entry point for merge experiments."""
    import argparse

    from dotenv import load_dotenv

    from .baseline import load_config

    load_dotenv()
    parser = argparse.ArgumentParser(description="Run LoRA merge experiment")
    parser.add_argument("--config", "-c", required=True, help="Path to config file")
    parser.add_argument(
        "--merge-type",
        "-m",
        choices=MERGE_TYPES,
        default=None,
        help="Merge strategy (default: merge.type in the config)",
    )
    parser.add_argument(
        "--approach",
        "-a",
        choices=["continual", "from_scratch"],
        default=None,
        help="Merging approach (default: merge.approach in the config)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_config(args.config)
    merge_type = args.merge_type or config.get("merge", {}).get("type", "ZipLoRA")
    log.info(f"Running {merge_type} merge experiment")
    results_df = run_merge_experiment(config, merge_type, args.approach)
    print(results_df.to_string())

    approach = args.approach or config.get("merge", {}).get("approach", "continual")
    results_path = (
        Path(os.getenv("PATHS_SAVE_DIR", "./saved_models")) / f"results_{merge_type}_{approach}.csv"
    )
    results_df.to_csv(results_path, index=False)
    log.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
