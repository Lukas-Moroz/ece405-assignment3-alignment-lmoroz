#!/bin/bash
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --job-name=a3_grpo
#SBATCH --output=logs/grpo_%j.log
#SBATCH --error=logs/grpo_%j.log
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16

REPO=/mnt/lustre/koa/scratch/lmoroz/ece405-assignment3-alignment-lmoroz
cd $REPO
mkdir -p logs results/grpo
source .venv/bin/activate

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start:   $(date)"

# --- NEW: Environment Verification Block ---
echo ""
echo "=== Environment Verification ==="
echo "Python Path: $(which python)"
python -c "import transformers; print('Transformers Version:', transformers.__version__)"
python -c "import vllm; print('vLLM Version:', vllm.__version__)"
echo "================================="
echo ""
# -------------------------------------------

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python experiments/run_grpo.py \
    --train-data $REPO/data/math/train.jsonl \
    --val-data $REPO/data/math/test.jsonl \
    --model-path $REPO/models/Qwen2.5-Math-1.5B \
    --output-dir $REPO/results/grpo \
    --n-grpo-steps 200 \
    --learning-rate 1e-5 \
    --rollout-batch-size 256 \
    --group-size 8 \
    --train-batch-size 256 \
    --gradient-accumulation-steps 128 \
    --gpu-memory-utilization 0.25 \
    --loss-type reinforce_with_baseline \
    --no-use-std-normalization \
    --eval-every 10 \
    --n-val 512

echo ""
echo "Done: $(date)"