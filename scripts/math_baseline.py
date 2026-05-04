"""
Problem (math_baseline): Zero-shot MATH evaluation with Qwen 2.5 Math 1.5B.

This script:
  1. Loads MATH validation examples from a JSONL file
  2. Formats prompts using the r1_zero prompt template
  3. Generates model outputs using vLLM
  4. Scores each output with r1_zero_reward_fn (format + answer rewards)
  5. Saves results to disk as JSONL for later analysis

Usage (on HPC/Colab where vllm is available):
    python scripts/math_baseline.py \
        --dataset_path /data/a5-alignment/MATH/validation.jsonl \
        --model_path /data/a5-alignment/models/Qwen2.5-Math-1.5B \
        --output_path outputs/math_baseline_results.jsonl

For local testing with the HuggingFace dataset:
    python scripts/math_baseline.py \
        --dataset_path data/math/test.jsonl \
        --model_path ../Qwen/Qwen2.5-0.5B \
        --output_path outputs/math_baseline_results.jsonl
"""

import argparse
import json
import os
from pathlib import Path
from typing import Callable, List

# vLLM is only available on Linux/HPC with Nvidia GPUs.
# If running locally for testing, this import will fail — that's expected.
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

# Paths
PROMPTS_DIR = Path(__file__).parent.parent / "cs336_alignment" / "prompts"


def load_prompt_template(name: str) -> str:
    """Load a prompt template from the prompts directory."""
    path = PROMPTS_DIR / f"{name}.prompt"
    return path.read_text()


def load_math_dataset(path: str) -> list[dict]:
    """Load MATH examples from a JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    print(f"Loaded {len(examples):,} examples from {path}")
    return examples


def format_prompts(examples: list[dict], prompt_template: str) -> list[str]:
    """Format each example as a string prompt using the given template."""
    prompts = []
    for ex in examples:
        prompt = prompt_template.format(question=ex["problem"])
        prompts.append(prompt)
    return prompts


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    examples: list[dict],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    output_path: str,
) -> dict:
    """
    Evaluate a language model on a list of prompts using vLLM,
    compute evaluation metrics using the reward function,
    and serialize results to disk.

    Args:
        vllm_model: Initialized vLLM LLM object.
        reward_fn: Callable that takes (model_response, ground_truth) and
                   returns a dict with keys: format_reward, answer_reward, reward.
        examples: Original dataset examples (must have 'answer' field).
        prompts: Formatted string prompts (parallel list with examples).
        eval_sampling_params: vLLM SamplingParams for generation.
        output_path: Path to write the JSONL results file.

    Returns:
        dict with aggregate evaluation metrics.
    """
    assert len(examples) == len(prompts), "examples and prompts must be same length"

    print(f"Generating responses for {len(prompts):,} prompts...")
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    total_format_reward = 0.0
    total_answer_reward = 0.0

    # Category counters for part (b)
    both_correct = 0       # format=1, answer=1
    format_only = 0        # format=1, answer=0
    both_zero = 0          # format=0, answer=0

    for example, prompt, output in zip(examples, prompts, outputs):
        generated_text = output.outputs[0].text
        # The prompt already contains "<think>", vLLM returns only the continuation.
        # The full response passed to reward_fn must include "</think> <answer>..." etc.
        full_response = generated_text

        ground_truth = example["answer"]
        rewards = reward_fn(full_response, ground_truth)

        fmt_r = rewards["format_reward"]
        ans_r = rewards["answer_reward"]
        total_format_reward += fmt_r
        total_answer_reward += ans_r

        if fmt_r == 1.0 and ans_r == 1.0:
            both_correct += 1
        elif fmt_r == 1.0 and ans_r == 0.0:
            format_only += 1
        else:
            both_zero += 1

        record = {
            "problem": example["problem"],
            "ground_truth": ground_truth,
            "level": example.get("level", ""),
            "type": example.get("type", ""),
            "prompt": prompt,
            "response": full_response,
            "format_reward": fmt_r,
            "answer_reward": ans_r,
            "reward": rewards["reward"],
        }
        results.append(record)

    n = len(results)
    metrics = {
        "n_examples": n,
        "both_correct (format=1, answer=1)": both_correct,
        "format_only (format=1, answer=0)": format_only,
        "both_zero (format=0, answer=0)": both_zero,
        "avg_format_reward": total_format_reward / n,
        "avg_answer_reward": total_answer_reward / n,
        "accuracy": both_correct / n,
    }

    # Print metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 60)

    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")
    print(f"\nResults saved to {output_path}")

    # Save metrics separately for easy reference
    metrics_path = output_path.replace(".jsonl", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot MATH baseline evaluation with Qwen 2.5 Math 1.5B"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data/a5-alignment/MATH/validation.jsonl",
        help="Path to the MATH validation JSONL file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/a5-alignment/models/Qwen2.5-Math-1.5B",
        help="Path to the model (HuggingFace format)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/math_baseline_results.jsonl",
        help="Path to write the results JSONL",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="If set, only evaluate on this many examples (useful for quick testing)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate per response",
    )
    args = parser.parse_args()

    # 1. Load dataset
    examples = load_math_dataset(args.dataset_path)
    if args.max_examples:
        examples = examples[: args.max_examples]
        print(f"Truncated to {len(examples):,} examples for quick testing")

    # 2. Format prompts using r1_zero template
    prompt_template = load_prompt_template("r1_zero")
    prompts = format_prompts(examples, prompt_template)

    # 3. Initialize vLLM model
    print(f"\nLoading model from {args.model_path}...")
    llm = LLM(model=args.model_path, dtype="bfloat16")

    # Sampling params as specified in the assignment handout
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        # Stop when the model closes the answer tag (per assignment spec)
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # 4 & 5. Generate, score, and serialize
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        examples=examples,
        prompts=prompts,
        eval_sampling_params=sampling_params,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
