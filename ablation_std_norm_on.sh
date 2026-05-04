#!/bin/bash
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --exclude=gn-07-09-00
#SBATCH --job-name=a3_grpo_std_norm_on
#SBATCH --output=logs/grpo_ablation_std_norm_on_%j.log
#SBATCH --error=logs/grpo_ablation_std_norm_on_%j.log
#SBATCH --mem=256G
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=16

cd /mnt/lustre/koa/scratch/lmoroz/ece405-assignment3-alignment-lmoroz
mkdir -p logs results/grpo_ablations/std_norm_on
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ID:    \$SLURM_JOB_ID"
echo "Node:      \$SLURMD_NODENAME"
echo "GPU:       \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Ablation:  std_norm_on"
echo "Start:     \$(date)"

python -u experiments/run_grpo_ablation.py \
    --train-data /mnt/lustre/koa/scratch/lmoroz/ece405-assignment3-alignment-lmoroz/data/math/train.jsonl \
    --val-data /mnt/lustre/koa/scratch/lmoroz/ece405-assignment3-alignment-lmoroz/data/math/test.jsonl \
    --model-path /mnt/lustre/koa/scratch/lmoroz/ece405-assignment3-alignment-lmoroz/models/Qwen2.5-Math-1.5B \
    --output-dir /mnt/lustre/koa/scratch/lmoroz/ece405-assignment3-alignment-lmoroz/results/grpo_ablations/std_norm_on \
    --n-grpo-steps 100 \
    --learning-rate 1e-5 \
    --rollout-batch-size 256 \
    --group-size 8 \
    --train-batch-size 256 \
    --gradient-accumulation-steps 128 \
    --gpu-memory-utilization 0.25 \
    --eval-every 10 \
    --n-val 256 \
    --loss-type reinforce_with_baseline --use-std-normalization --prompt-name r1_zero --length-norm mean

echo "Done: \$(date)"
