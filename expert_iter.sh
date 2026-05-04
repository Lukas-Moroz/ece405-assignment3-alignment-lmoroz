#!/bin/bash
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:1
#SBATCH --constraint="hopper"
#SBATCH --job-name=a3_expert_iter
#SBATCH --output=logs/expert_iter_%j.log
#SBATCH --error=logs/expert_iter_%j.log
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16

# Main Assignment §5: Expert Iteration on MATH
# Two configs as required: G=4 and G=8 rollouts per question
# vLLM and policy share single A4000 GPU (gpu_memory_utilization=0.45 each)
# Output: checkpoints/expert_iter/*, results/expert_iter/*.json


REPO=/mnt/lustre/koa/scratch/lmoroz/ece405-assignment3-alignment-lmoroz
cd $REPO
mkdir -p logs results/expert_iter checkpoints/expert_iter
source .venv/bin/activate

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start:   $(date)"

if [ ! -f $REPO/data/math/train.jsonl ]; then
    echo "ERROR: data/math/train.jsonl not found — run job0_data_prep first"
    exit 1
fi

# Config 1: G=4, batch 512, 5 steps
python experiments/run_expert_iter.py \
    --train-data $REPO/data/math/train.jsonl \
    --val-data   $REPO/data/math/test.jsonl \
    --n-ei-steps 5 \
    --batch-size 512 \
    --group-size 4 \
    --sft-steps  40

# Config 2: G=8, batch 1024, 5 steps
python experiments/run_expert_iter.py \
    --train-data $REPO/data/math/train.jsonl \
    --val-data   $REPO/data/math/test.jsonl \
    --n-ei-steps 5 \
    --batch-size 1024 \
    --group-size 8 \
    --sft-steps  40

echo ""
echo "Done: $(date)"