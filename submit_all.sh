#!/bin/bash
#
# submit_all.sh — Orchestrate all assignment jobs with Slurm dependencies
#
# Job dependency chain:
#   data_prep.sh → (math_baseline.sh, sft_math.sh, expert_iter.sh)
#
# Usage:
#   bash submit_all.sh
#
# The script will output job IDs and wait for them to complete.

set -e

REPO=/mnt/lustre/koa/scratch/lmoroz/ece405-assignment3-alignment-lmoroz
cd "$REPO"

echo "Submitting ECE405 Alignment Assignment Jobs"

# Create log directory
mkdir -p logs

# Job 0: Data Prep (no dependency)
echo "[0/3] Submitting data_prep.sh (downloading MATH dataset)..."
JOB0=$(sbatch data_prep.sh | awk '{print $NF}')
echo "      Job ID: $JOB0"
echo ""

# Job 1: Math Baseline (depends on data_prep)
echo "[1/3] Submitting math_baseline.sh (zero-shot evaluation)..."
echo "      Dependency: afterok:$JOB0"
JOB1=$(sbatch --dependency=afterok:$JOB0 math_baseline.sh | awk '{print $NF}')
echo "      Job ID: $JOB1"
echo ""

# Job 2: SFT Math (depends on data_prep)
echo "[2/3] Submitting sft_math.sh (SFT dataset size sweep + filtered)..."
echo "      Dependency: afterok:$JOB0"
JOB2=$(sbatch --dependency=afterok:$JOB0 sft_math.sh | awk '{print $NF}')
echo "      Job ID: $JOB2"
echo ""

# Job 3: Expert Iteration (depends on data_prep)
echo "[3/3] Submitting expert_iter.sh (Expert iteration G=4,8)..."
echo "      Dependency: afterok:$JOB0"
JOB3=$(sbatch --dependency=afterok:$JOB0 expert_iter.sh | awk '{print $NF}')
echo "      Job ID: $JOB3"
echo ""

echo "All jobs submitted successfully!"
echo "Job Summary:"
echo "  [0] Data Prep       (CPU):  $JOB0"
echo "  [1] Math Baseline   (GPU):  $JOB1  → depends on $JOB0"
echo "  [2] SFT Sweep       (GPU):  $JOB2  → depends on $JOB0"
echo "  [3] Expert Iter     (GPU):  $JOB3  → depends on $JOB0"
echo ""
echo "Monitor all jobs:"
echo "  squeue -u $USER"
echo ""
echo "View logs:"
echo "  ls -lh logs/"
echo ""
