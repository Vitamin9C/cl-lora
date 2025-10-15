#!/bin/bash
#SBATCH --job-name=cl_lora_baseline
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# CL-LoRA Baseline Benchmark SLURM Script
# Runs baseline continual learning experiments (replay or no_replay)

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
CONFIG_FILE="${1:-configs/experiments/baseline_spectral.yaml}"
EXPERIMENT_TYPE="${2:-no_replay}"  # Options: replay, no_replay

echo "============================================"
echo "CL-LoRA Baseline Experiment"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Config: $CONFIG_FILE"
echo "Type: $EXPERIMENT_TYPE"
echo "Start time: $(date)"
echo "============================================"

# Run experiment
python -m cl_lora.experiments.baseline \
    --config "$CONFIG_FILE" \
    --type "$EXPERIMENT_TYPE"

echo "============================================"
echo "End time: $(date)"
echo "============================================"
