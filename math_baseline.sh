#!/bin/bash
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:1
#SBATCH --job-name=a3_math_baseline
#SBATCH --output=logs/math_baseline_%j.log
#SBATCH --error=logs/math_baseline_%j.log
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=16

# Main Assignment §3: Zero-shot MATH baseline
# Evaluates Qwen2.5-Math-1.5B on data/math/test.jsonl
# Output: results/math_baseline/results.json


REPO=/mnt/lustre/koa/scratch/lmoroz/ece405-assignment3-alignment-lmoroz
cd $REPO
mkdir -p logs results/math_baseline
source .venv/bin/activate

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start:   $(date)"

python scripts/math_baseline.py \
    --model_path  $REPO/models/Qwen2.5-Math-1.5B \
    --dataset_path $REPO/data/math/test.jsonl \
    --output_path  $REPO/results/math_baseline/results.jsonl

echo ""
echo "Done: $(date)"