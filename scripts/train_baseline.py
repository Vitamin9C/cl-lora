#!/usr/bin/env python
"""Run baseline continual learning experiment.

Usage:
    python scripts/train_baseline.py --config configs/experiments/baseline_spectral.yaml
    python scripts/train_baseline.py --config configs/experiments/baseline_spectral.yaml --type replay
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cl_lora.experiments.baseline import main

if __name__ == "__main__":
    main()
