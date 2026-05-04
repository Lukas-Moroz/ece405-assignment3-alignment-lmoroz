#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=a3_data_prep
#SBATCH --output=logs/data_prep_%j.log
#SBATCH --error=logs/data_prep_%j.log
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4

# No GPU needed — downloads MATH dataset and creates SFT format data.
# All downstream GPU jobs depend on this finishing first.

cd /mnt/lustre/koa/scratch/lmoroz/ece405-assignment3-alignment-lmoroz
mkdir -p logs data/math

source .venv/bin/activate

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Start:   $(date)"

# Step 1: Download MATH dataset → data/math/train.jsonl + data/math/test.jsonl
echo ""
echo "--- Downloading MATH dataset ---"
python download_math_dataset.py

# Verify
echo "train.jsonl: $(wc -l < data/math/train.jsonl) examples"
echo "test.jsonl:  $(wc -l < data/math/test.jsonl) examples"

# Step 2: Create SFT data from competition solutions
# Wraps each problem+solution into r1_zero prompt/response format
echo ""
echo "--- Creating SFT training data ---"
python - << 'PYEOF'
import json
from pathlib import Path

PROMPT = (
    "A conversation between User and Assistant. The User asks a question, "
    "and the Assistant solves it. The reasoning process is enclosed within "
    "<think> </think> and answer is enclosed within <answer> </answer> tags, "
    "i.e., <think> reasoning process here </think><answer> answer here </answer>.\n\n"
    "User: {question}\nAssistant: <think>"
)

train_path = Path("data/math/train.jsonl")
sft_path   = Path("data/math/sft.jsonl")

written = skipped = 0
with open(train_path) as fin, open(sft_path, "w") as fout:
    for line in fin:
        ex = json.loads(line)
        answer  = ex.get("answer", "").strip()
        problem = ex.get("problem", "")
        solution = ex.get("solution", "")
        if not answer or not problem:
            skipped += 1
            continue
        record = {
            "prompt":   PROMPT.format(question=problem),
            "response": f"{solution}</think><answer>{answer}</answer>",
            "answer":   answer,
            "problem":  problem,
        }
        fout.write(json.dumps(record) + "\n")
        written += 1

print(f"SFT data: {written} written, {skipped} skipped -> data/math/sft.jsonl")
PYEOF

echo ""
echo "Data ready:"
ls -lh data/math/
echo ""
echo "Done: $(date)"