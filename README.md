# CL-LoRA: Continual Learning with Low-Rank Adaptation

Continual learning experiments using LoRA adapters on pretrained Vision Transformers for multi-label land cover classification on BigEarthNet-V2.

## Features

- **Two backbone models**: SpectralGPT and SoftCon (DINOv2-based)
- **Multiple CL strategies**: Replay, No-Replay
- **LoRA merging**: ZipLoRA, LoRASoups, LoRAHub
- **BigEarthNet-V2**: Multi-label Sentinel-2 classification across European countries

## Attribution

This project builds upon and incorporates work from:

- **SpectralGPT**: [SpectralGPT GitHub](https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT)
  - Hong, D., et al. "SpectralGPT: Spectral Remote Sensing Foundation Model." IEEE TPAMI, 2024.
- **SoftCon**: [SoftCon GitHub](https://github.com/zhu-xlab/softcon)
  - Wang, Y., et al. "SSL4EO-S12: A Large-Scale Multi-Modal, Multi-Temporal Dataset for Self-Supervised Learning in Earth Observation." CVPRW, 2023.
- **LoRA-ViT**: Adapted from [JamesQFreeman/LoRA-ViT](https://github.com/JamesQFreeman/LoRA-ViT)

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

### 2. Run baseline experiment

```bash
python scripts/train_baseline.py --config configs/experiments/baseline_spectral.yaml
```

### 3. Run LoRA merging experiment

```bash
python scripts/train_merge.py --config configs/experiments/merge_ziplora.yaml
```

## Project Structure

```
cl-lora/
├── src/cl_lora/          # Main package
│   ├── models/           # ViT, LoRA, backbone wrappers
│   ├── data/             # Datasets, transforms, constants
│   └── training/         # Lightning modules, metrics
├── configs/              # YAML configuration files
├── scripts/              # CLI entry points
├── notebooks/            # Analysis notebooks
├── baseline/             # Original baseline code (legacy)
└── merge/                # Original merge code (legacy)
```

## Configuration

All experiments are configured via YAML files. See `configs/` for examples.

Key configuration sections:

- `model`: Backbone type, LoRA rank, number of classes
- `data`: Dataset paths, batch size, augmentation settings
- `training`: Learning rate, epochs, early stopping
- `experiment`: CL strategy, countries, sample sizes

## License

MIT License - See [LICENSE](LICENSE) for details.

Note: This repository uses code adapted from SpectralGPT, SoftCon, and LoRA-ViT projects.
Please refer to their respective licenses for terms of use.
