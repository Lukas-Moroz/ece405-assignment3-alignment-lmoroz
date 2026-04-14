"""
Download the MATH dataset from HuggingFace and save as JSONL files
under data/math/ in the format expected by the evaluation script.

Uses huggingface_hub to directly download parquet files.

Run with:
    uv run --no-sync python download_math_dataset.py
"""
import json
import os

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

import re

OUTPUT_DIR = "data/math"


def extract_boxed_answer(solution: str) -> str:
    """
    Extract the answer from the last \\boxed{...} in the solution string.
    Handles nested braces.
    """
    # Find all \boxed{ occurrences, take the last one
    idx = solution.rfind(r"\boxed{")
    if idx == -1:
        idx = solution.rfind(r"\boxed {")
    if idx == -1:
        return ""

    # Walk forward counting braces to find the matching closing brace
    start = solution.index("{", idx)
    depth = 0
    for i, ch in enumerate(solution[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return solution[start + 1 : i]
    return ""

os.makedirs(OUTPUT_DIR, exist_ok=True)

REPO_ID = "qwedsacf/competition_math"

# Find all parquet files in the repo
all_files = list(list_repo_files(REPO_ID, repo_type="dataset"))
parquet_files = [f for f in all_files if f.endswith(".parquet")]
print(f"Found parquet files: {parquet_files}")

# This dataset only has a train split — we'll make our own train/test split
dfs = []
for parquet_path in parquet_files:
    print(f"\nDownloading {parquet_path}...")
    local_file = hf_hub_download(
        repo_id=REPO_ID,
        filename=parquet_path,
        repo_type="dataset",
    )
    df = pd.read_parquet(local_file)
    print(f"  Columns: {list(df.columns)}")
    print(f"  Rows: {len(df):,}")
    dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)
print(f"\nTotal rows: {len(all_data):,}")

# Shuffle and split into train (80%) and test (20%)
all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)
split_idx = int(len(all_data) * 0.8)
splits = {
    "train": all_data.iloc[:split_idx],
    "test": all_data.iloc[split_idx:],
}

for split_name, df_split in splits.items():
    out_path = os.path.join(OUTPUT_DIR, f"{split_name}.jsonl")
    with open(out_path, "w") as f:
        for _, row in df_split.iterrows():
            solution_text = str(row.get("solution", ""))
            record = {
                "problem": row.get("problem", row.get("question", "")),
                "solution": solution_text,
                "answer": extract_boxed_answer(solution_text),   # extracted short answer
                "level": str(row.get("level", "")),
                "type": str(row.get("type", "")),
            }
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(df_split):,} examples -> {out_path}")

print("\nDone!")
