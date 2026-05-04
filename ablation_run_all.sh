#!/bin/bash
# Submit all §8 GRPO ablation runs.
# Usage:
#   bash ablation_run_all.sh              # submit immediately, no dependency
#   bash ablation_run_all.sh 12459142     # wait for job 12459142 to finish first

DEPENDENCY_JOB=$1
REPO=/mnt/lustre/koa/scratch/lmoroz/ece405-assignment3-alignment-lmoroz
cd $REPO
mkdir -p $REPO/results/grpo_ablations $REPO/logs

if [ -n "$DEPENDENCY_JOB" ]; then
    DEP_FLAG="--dependency=afterany:${DEPENDENCY_JOB}"
    echo "Chaining ablations to wait for job ${DEPENDENCY_JOB}..."
else
    DEP_FLAG=""
    echo "Submitting ablations immediately (no dependency)..."
fi

submit() {
    local NAME=$1
    shift
    local EXTRA_ARGS="$@"
    local SCRIPT="ablation_${NAME}.sh"

    cat > $SCRIPT <<INNER_EOF
#!/bin/bash
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --exclude=gn-07-09-00
#SBATCH --job-name=a3_grpo_${NAME}
#SBATCH --output=logs/grpo_ablation_${NAME}_%j.log
#SBATCH --error=logs/grpo_ablation_${NAME}_%j.log
#SBATCH --mem=256G
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=16

cd $REPO
mkdir -p logs results/grpo_ablations/${NAME}
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ID:    \\\$SLURM_JOB_ID"
echo "Node:      \\\$SLURMD_NODENAME"
echo "GPU:       \\\$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Ablation:  ${NAME}"
echo "Start:     \\\$(date)"

python -u experiments/run_grpo_ablation.py \\
    --train-data $REPO/data/math/train.jsonl \\
    --val-data $REPO/data/math/test.jsonl \\
    --model-path $REPO/models/Qwen2.5-Math-1.5B \\
    --output-dir $REPO/results/grpo_ablations/${NAME} \\
    --n-grpo-steps 100 \\
    --learning-rate 1e-5 \\
    --rollout-batch-size 256 \\
    --group-size 8 \\
    --train-batch-size 256 \\
    --gradient-accumulation-steps 128 \\
    --gpu-memory-utilization 0.25 \\
    --eval-every 10 \\
    --n-val 256 \\
    ${EXTRA_ARGS}

echo "Done: \\\$(date)"
INNER_EOF
    chmod +x $SCRIPT
    sbatch $DEP_FLAG $SCRIPT
}

submit no_baseline       --loss-type no_baseline                 --no-use-std-normalization --prompt-name r1_zero       --length-norm mean
submit std_norm_on       --loss-type reinforce_with_baseline     --use-std-normalization    --prompt-name r1_zero       --length-norm mean
submit question_only     --loss-type reinforce_with_baseline     --no-use-std-normalization --prompt-name question_only --length-norm mean
submit length_normalize  --loss-type reinforce_with_baseline     --no-use-std-normalization --prompt-name r1_zero       --length-norm normalize

echo ""
echo "Check status with: squeue -u lmoroz"
