#!/bin/bash
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --job-name=a3_grpo_smoke
#SBATCH --output=logs/grpo_smoke_%j.log
#SBATCH --error=logs/grpo_smoke_%j.log
#SBATCH --mem=128G
#SBATCH --time=30:00
#SBATCH --cpus-per-task=8

REPO=/mnt/lustre/koa/scratch/lmoroz/ece405-assignment3-alignment-lmoroz
cd $REPO
mkdir -p logs results/grpo_smoke
source .venv/bin/activate

echo "Start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python experiments/run_grpo.py \
    --train-data $REPO/data/math/train.jsonl \
    --val-data $REPO/data/math/test.jsonl \
    --output-dir $REPO/results/grpo_smoke \
    --n-grpo-steps 2 \
    --eval-every 2 \
    --n-val 20 \
    --rollout-batch-size 32 \
    --group-size 4 \
    --train-batch-size 32 \
    --gradient-accumulation-steps 16

echo "Done: $(date)"
