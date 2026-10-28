#!/bin/bash
# Quick run script for merge experiments (local execution)
# Usage: ./scripts/run_merge.sh [config_file] [ZipLoRA|LoRASoups|LoRAHub] [continual|from_scratch]

set -e

CONFIG="${1:-configs/experiments/merge_ziplora.yaml}"
MERGE_TYPE="${2:-ZipLoRA}"
APPROACH="${3:-continual}"

echo "Running merge experiment..."
echo "Config: $CONFIG"
echo "Merge type: $MERGE_TYPE"
echo "Approach: $APPROACH"

python -m cl_lora.experiments.merge --config "$CONFIG" --merge-type "$MERGE_TYPE" --approach "$APPROACH"
