#!/usr/bin/env python
"""Run LoRA merge experiment for continual learning.

Usage:
    python scripts/train_merge.py --config configs/experiments/merge_ziplora.yaml
    python scripts/train_merge.py --config configs/experiments/merge_ziplora.yaml --merge-type LoRAHub
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cl_lora.experiments.merge import main

if __name__ == "__main__":
    main()
