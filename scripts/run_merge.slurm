#!/bin/bash
#SBATCH --job-name=cl_lora_merge
#SBATCH --output=logs/merge_%j.out
#SBATCH --error=logs/merge_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# CL-LoRA Merge Experiment SLURM Script
# Runs LoRA merging experiments (ZipLoRA, LoRAHub, LoRASoups)

# Load environment
source ~/.bashrc
eval "$(micromamba shell hook --shell bash)"
micromamba activate cl_lora_env

# Create logs directory if not exists
mkdir -p logs

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Set environment variables
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH}"

# Configuration
CONFIG_FILE="${1:-configs/experiments/merge_ziplora.yaml}"
MERGE_TYPE="${2:-ZipLoRA}"  # Options: ZipLoRA, LoRAHub, LoRASoups

echo "============================================"
echo "CL-LoRA Merge Experiment"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Config: $CONFIG_FILE"
echo "Merge Type: $MERGE_TYPE"
echo "Start time: $(date)"
echo "============================================"

# Run experiment
python -m cl_lora.experiments.merge \
    --config "$CONFIG_FILE" \
    --merge-type "$MERGE_TYPE"

echo "============================================"
echo "End time: $(date)"
echo "============================================"
