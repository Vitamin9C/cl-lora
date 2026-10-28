# CL-LoRA: Continual Learning with Low-Rank Adaptation

Continual learning experiments using LoRA adapters on pretrained Vision Transformers for multi-label land cover classification on BigEarthNet-V2. Countries arrive one after another (domain-incremental learning: same 19 classes, different data distribution); the question is how well LoRA merging keeps old countries while adding new ones.

## Features

- **Two backbone models**: SpectralGPT and SoftCon (DINOv2-based)
- **Baselines**: Replay (retrain on all seen data, upper bound), No replay (sequential fine-tuning, lower bound), Task tuning (one independent LoRA per country)
- **LoRA merging** of the task-tuned heads: ZipLoRA, LoRA Soups, LoRAHub, each **continual** (merge the previous result with the next country, with a replay memory) or **from scratch** (merge all seen countries at once)
- **Metrics**: micro / macro average precision per country, pooled and mean; forgetting and intransigence (`cl_lora.training.forgetting_and_intransigence`)
- **BigEarthNet-V2**: Multi-label Sentinel-2 classification across European countries; datasets can be saved once to run on a machine without the LMDB

## Attribution

This project builds upon and incorporates work from:

- **SpectralGPT**: [SpectralGPT GitHub](https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT)
  - Hong, D., et al. "SpectralGPT: Spectral Remote Sensing Foundation Model." IEEE TPAMI, 2024.
- **SoftCon**: [SoftCon GitHub](https://github.com/zhu-xlab/softcon)
  - Wang, Y., et al. "Multi-Label Guided Soft Contrastive Learning for Efficient Earth Observation Pretraining." IEEE TGRS, 2024.
- **LoRA-ViT**: Adapted from [JamesQFreeman/LoRA-ViT](https://github.com/JamesQFreeman/LoRA-ViT)
- **ZipLoRA**: Shah, V., et al. "ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs." ECCV 2024.
- **LoRAHub**: Huang, C., et al. "LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition." COLM 2024.

The experiments were developed in the team repository [arnemaass/cl-lora](https://github.com/arnemaass/cl-lora); this repository is its refactored package version.

## Installation

### Using uv (recommended)

```bash
uv sync
```

### Using pip

```bash
pip install -r requirements.txt
pip install -e .
```

`requirements.lock.txt` lists the exact environment of the cluster runs (Python 3.11.9).

### External Dependencies

SoftCon backbone requires the softcon package:

```bash
pip install git+https://github.com/zhu-xlab/softcon.git
```

## Quick Start

### 1. Configure paths

Copy the example environment file and fill in your local paths:

```bash
cp .env.example .env
# Edit .env with your data and model paths
```

### 2. Run the baselines

```bash
python scripts/train_baseline.py --config configs/experiments/baseline_spectral.yaml          # no replay
python scripts/train_baseline.py --config configs/experiments/baseline_replay_spectral.yaml   # replay
```

### 3. Train the task heads

```bash
python scripts/train_baseline.py --config configs/experiments/task_tuning_spectral.yaml
```

This saves `<country>_lora.safetensors` and `<country>_fc.safetensors` per country under `$PATHS_SAVE_DIR/task_tuning/permutation_<...>/`. Point `PATHS_TASK_HEADS` (or `merge.task_heads_dir`) to that folder.

### 4. Run LoRA merging

```bash
python scripts/train_merge.py --config configs/experiments/merge_ziplora.yaml
python scripts/train_merge.py --config configs/experiments/merge_soups.yaml --approach from_scratch
python scripts/train_merge.py --config configs/experiments/merge_lorahub.yaml
```

### Running on another machine

See [docs/mobility_setup.md](docs/mobility_setup.md) for copying the task heads and using saved datasets (`scripts/save_datasets.py`).

## Merging strategies

All strategies merge per LoRA target (the q and v projections of every block), apply the merged delta to the frozen backbone, and build the classifier as a weighted average of the heads' classifiers (continual: old (t-1)/t, new 1/t), which is then trained on the merge data.

| Strategy | Merged delta for one target | Trained / searched |
|---|---|---|
| ZipLoRA | sum over heads of m_h (one value per input column) times Delta_h | merger vectors m with task loss + overlap penalty abs(cos(m_1, m_2)) |
| LoRA Soups | sum over heads of softmax(alpha)_h times Delta_h | one mixing weight per head and target |
| LoRAHub | (sum_h w_h B_h)(sum_h w_h A_h), scaled by alpha / r | one weight per head, Nevergrad search (init 0, range [-1.5, 1.5], L1 0.05) |

Delta_h = (alpha / r) B_h A_h is the delta of head h.

## Results

SpectralGPT, countries Finland → Ireland → Serbia → Portugal, 5,000 train / 500 test samples per country, LoRA rank 4, replay memory 500. Mean micro AP over the seen countries after the last step (logs in `logs/`):

| Method | Mean micro AP |
|---|---|
| Replay (upper bound) | 0.737 |
| ZipLoRA (continual) | 0.588 |
| LoRAHub (continual) | 0.574 |
| LoRA Soups (continual, 60 epochs) | 0.562 |
| No replay (lower bound) | 0.479 |

Task tuning alone reaches 0.853 / 0.840 / 0.748 / 0.602 micro AP on Finland / Ireland / Serbia / Portugal.

**The merge numbers above come from the original code in the team repository, which had two bugs that this version fixes** (see `logs/` for the raw output):

- ZipLoRA and LoRA Soups applied saved LoRA key `i` to transformer block `i`, although the keys alternate q and v per block (`i = 2 * block + {0: q, 1: v}`). Most deltas went to the wrong block or projection, and the LoRAs of blocks 6–11 were not used.
- LoRAHub passed the task LoRAs to PEFT with their saved key names, which PEFT did not recognise (`PEFT load unexpected: ['w_a_000', ...]` in the logs), so every merged LoRA had a zero delta.

The merge results therefore mainly measure the classifier retrained on the replay memory on top of the pretrained backbone. Rerun the merges with this version for LoRA-merging results.

## Changes relative to the original code

- LoRA targets are matched to saved keys in creation order and checked against the wrapper's `w_As` (`cl_lora.models.lora_heads.lora_targets`).
- Deltas include the LoRA scaling alpha / r; the scaling uses float division.
- Per-country evaluation uses the dataset position in the permutation (the original fix for non-identity permutations; affected logs are in `logs/baseline/epoch15/wrong_eval/`).
- LoRAHub searches the head weights on replay data plus an equally sized sample of new data, then trains the classifier on the merged backbone.
- Saved datasets keep a separate test split and are stored without augmentation.

## Project Structure

```
cl-lora/
├── src/cl_lora/          # Main package
│   ├── models/           # ViT, LoRA, backbone wrappers, LoRA heads
│   ├── data/             # Datasets, transforms, constants, saved datasets
│   ├── training/         # Lightning modules, metrics, trainer factory
│   └── experiments/      # Baseline and merge runners, merging strategies
├── configs/              # YAML configuration files
├── scripts/              # CLI entry points and SLURM scripts
├── notebooks/            # Analysis and plotting notebooks
├── logs/                 # Logs of the reported runs (baseline/, merge/)
└── docs/                 # Setup notes
```

## Configuration

All experiments are configured via YAML files. See `configs/` for examples.

Key configuration sections:

- `model`: Backbone type, LoRA rank and alpha, number of classes
- `data`: Batch size, workers, optional saved-datasets directory
- `training`: Learning rate, epochs, optional early stopping
- `experiment`: Baseline type, countries and order, sample sizes, replay memory
- `merge`: Merge strategy and approach, task-heads directory, strategy settings

## License

MIT License - See [LICENSE](LICENSE) for details.

Note: This repository uses code adapted from SpectralGPT, SoftCon, and LoRA-ViT projects.
Please refer to their respective licenses for terms of use.
