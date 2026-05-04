#!/bin/bash
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --exclude=gn-07-09-00
#SBATCH --job-name=a3_sft_math
#SBATCH --output=logs/sft_math_%j.log
#SBATCH --error=logs/sft_math_%j.log
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16

REPO=/mnt/lustre/koa/scratch/lmoroz/ece405-assignment3-alignment-lmoroz
cd $REPO
mkdir -p logs results/sft_math checkpoints/sft_math
source .venv/bin/activate

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start:   $(date)"

echo "--- Dataset size sweep ---"
python -u experiments/run_sft_math.py \
    --mode sweep \
    --sft-data $REPO/data/math/sft.jsonl \
    --val-data $REPO/data/math/test.jsonl \
    --model-path $REPO/models/Qwen2.5-Math-1.5B \
    --n-steps 400 \
    --eval-every 80 \
    --n-val-examples 200 \
    --lr 2e-5 \
    --grad-accum 8

echo ""
echo "--- Filtered SFT ---"
python -u experiments/run_sft_math.py \
    --mode filtered \
    --sft-data $REPO/data/math/sft.jsonl \
    --val-data $REPO/data/math/test.jsonl \
    --model-path $REPO/models/Qwen2.5-Math-1.5B \
    --n-steps 400 \
    --eval-every 80 \
    --n-val-examples 200 \
    --lr 2e-5 \
    --grad-accum 8

echo ""
echo "Done: $(date)"