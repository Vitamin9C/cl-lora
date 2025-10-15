#!/bin/bash
# Quick run script for baseline experiments (local execution)
# Usage: ./scripts/run_baseline.sh [config_file] [experiment_type]

set -e

CONFIG="${1:-configs/experiments/baseline_spectral.yaml}"
TYPE="${2:-no_replay}"

echo "Running baseline experiment..."
echo "Config: $CONFIG"
echo "Type: $TYPE"

python -m cl_lora.experiments.baseline --config "$CONFIG" --type "$TYPE"
