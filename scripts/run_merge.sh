#!/bin/bash
# Quick run script for merge experiments (local execution)
# Usage: ./scripts/run_merge.sh [config_file] [merge_type]

set -e

CONFIG="${1:-configs/experiments/merge_ziplora.yaml}"
MERGE_TYPE="${2:-ZipLoRA}"

echo "Running merge experiment..."
echo "Config: $CONFIG"
echo "Merge type: $MERGE_TYPE"

python -m cl_lora.experiments.merge --config "$CONFIG" --merge-type "$MERGE_TYPE"
