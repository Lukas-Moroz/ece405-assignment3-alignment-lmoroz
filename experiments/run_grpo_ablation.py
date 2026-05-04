"""
GRPO training loop on MATH dataset.

Usage:
    python experiments/run_grpo.py \
        --train-data data/math/train.jsonl \
        --val-data data/math/test.jsonl \
        --output-dir results/grpo
"""
from __future__ import annotations

import os
# Disable Triton MMA kernels on Hopper — avoids mma->mma Ampere assertion
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from vllm import LLM, SamplingParams
from unittest.mock import patch

from cs336_alignment.grpo import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
    masked_mean,
)
from cs336_alignment.sft import (
    tokenize_prompt_and_output,
    get_response_log_probs,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

REPO = Path(__file__).parent.parent

# Data loading
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_prompt(name):
    return (REPO / "cs336_alignment" / "prompts" / f"{name}.prompt").read_text()

# vLLM setup (from handout)
def init_vllm(model_id, device, seed, gpu_memory_utilization=0.45):
    from vllm.model_executor import set_random_seed as vllm_set_random_seed
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            enforce_eager=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm(policy, llm):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

# Evaluation
def evaluate(llm, val_examples, prompt_template, reward_fn, sampling_params, n_val=1024):
    """Return mean answer reward on validation subset."""
    val_subset = val_examples[:n_val]
    prompts = [prompt_template.format(question=ex["problem"]) for ex in val_subset]
    gts = [ex["answer"] for ex in val_subset]

    outputs = llm.generate(prompts, sampling_params)
    rewards = []
    for out, gt in zip(outputs, gts):
        resp = out.outputs[0].text
        r = reward_fn(resp, gt)
        rewards.append(r["reward"])
    return sum(rewards) / len(rewards), rewards

# Main train loop
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", required=True)
    p.add_argument("--val-data", required=True)
    p.add_argument("--model-path", default=str(REPO / "models" / "Qwen2.5-Math-1.5B"))
    p.add_argument("--output-dir", default=str(REPO / "results" / "grpo"))

    # GRPO hyperparams (matches handout defaults)
    p.add_argument("--n-grpo-steps", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--advantage-eps", type=float, default=1e-6)
    p.add_argument("--rollout-batch-size", type=int, default=256)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--sampling-temperature", type=float, default=1.0)
    p.add_argument("--sampling-min-tokens", type=int, default=4)
    p.add_argument("--sampling-max-tokens", type=int, default=1024)
    p.add_argument("--epochs-per-rollout-batch", type=int, default=1)
    p.add_argument("--train-batch-size", type=int, default=256)
    p.add_argument("--gradient-accumulation-steps", type=int, default=128)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    p.add_argument("--loss-type", choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"],
                   default="reinforce_with_baseline")
    p.add_argument("--use-std-normalization", action=argparse.BooleanOptionalAction, default=False)    
    p.add_argument("--cliprange", type=float, default=0.2)
    p.add_argument("--prompt-name", default="r1_zero", choices=["r1_zero", "question_only"])
    p.add_argument("--length-norm", default="mean", choices=["mean", "normalize"])

    # Eval
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--n-val", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--policy-device", default="cuda:0")
    p.add_argument("--vllm-device", default="cuda:0")  # sharing GPU on A4000
    args = p.parse_args()

    # Asserts from handout
    assert args.train_batch_size % args.gradient_accumulation_steps == 0
    micro_train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    assert args.rollout_batch_size % args.group_size == 0
    n_prompts_per_rollout = args.rollout_batch_size // args.group_size

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / "metrics.jsonl"
    metrics_file = open(metrics_path, "w")

    def log(d):
        metrics_file.write(json.dumps(d) + "\n")
        metrics_file.flush()

    # Load data
    print(f"Loading train data from {args.train_data}")
    train_examples = load_jsonl(args.train_data)
    print(f"Loading val data from {args.val_data}")
    val_examples = load_jsonl(args.val_data)

    prompt_template = load_prompt(args.prompt_name)

    # Load tokenizer + policy
    print(f"Loading tokenizer/policy from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float32, trust_remote_code=True
    ).to(args.policy_device)
    policy.train()

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=20,
        num_training_steps=args.n_grpo_steps,
    )

    # Init vLLM for rollouts
    print("Initializing vLLM...")
    llm = init_vllm(args.model_path, args.vllm_device, args.seed, args.gpu_memory_utilization)

    rollout_sampling = SamplingParams(
        temperature=args.sampling_temperature,
        top_p=1.0,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens,
        n=args.group_size,
        seed=args.seed,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    eval_sampling = SamplingParams(
        temperature=args.sampling_temperature,
        top_p=1.0,
        max_tokens=args.sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # Training loop
    for step in range(1, args.n_grpo_steps + 1):
        t_step = time.time()

        # Sample batch of prompts
        batch = random.sample(train_examples, n_prompts_per_rollout)
        questions = [ex["problem"] for ex in batch]
        gts = [ex["answer"] for ex in batch]

        prompts = [prompt_template.format(question=q) for q in questions]
        repeated_gts = [gt for gt in gts for _ in range(args.group_size)]

        # Sync weights to vLLM and rollout
        load_policy_into_vllm(policy, llm)
        t_rollout = time.time()
        outputs = llm.generate(prompts, rollout_sampling)
        rollout_responses = []
        for out in outputs:
            for o in out.outputs:
                rollout_responses.append(o.text)
        assert len(rollout_responses) == args.rollout_batch_size
        rollout_time = time.time() - t_rollout

        # Compute advantages
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_gts,
            group_size=args.group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.use_std_normalization,
        )

        # Tokenize prompts and responses
        flat_prompts = []
        for p_str in prompts:
            flat_prompts.extend([p_str] * args.group_size)

        tok = tokenize_prompt_and_output(flat_prompts, rollout_responses, tokenizer)
        input_ids = tok["input_ids"].to(args.policy_device)
        labels = tok["labels"].to(args.policy_device)
        response_mask = tok["response_mask"].to(args.policy_device)
        advantages_dev = advantages.to(args.policy_device).unsqueeze(1)  # (B, 1)
        raw_rewards_dev = raw_rewards.to(args.policy_device).unsqueeze(1)

        # Off-policy: compute old log-probs once
        off_policy = args.loss_type == "grpo_clip" or args.epochs_per_rollout_batch > 1
        old_log_probs = None
        if off_policy:
            with torch.inference_mode():
                old_log_probs_full = []
                for i in range(0, input_ids.size(0), micro_train_batch_size):
                    mb_ids = input_ids[i:i + micro_train_batch_size]
                    mb_lab = labels[i:i + micro_train_batch_size]
                    out = get_response_log_probs(policy, mb_ids, mb_lab, return_token_entropy=False)
                    old_log_probs_full.append(out["log_probs"])
                old_log_probs = torch.cat(old_log_probs_full, dim=0).detach()

        # Training phase
        optimizer.zero_grad()
        total_loss = 0.0
        total_entropy = 0.0
        clip_fractions = []
        n_micro = 0

        for epoch in range(args.epochs_per_rollout_batch):
            # Shuffle indices
            perm = torch.randperm(input_ids.size(0))
            for start in range(0, input_ids.size(0), micro_train_batch_size):
                idx = perm[start:start + micro_train_batch_size]
                mb_ids = input_ids[idx]
                mb_lab = labels[idx]
                mb_mask = response_mask[idx]
                mb_adv = advantages_dev[idx]
                mb_raw = raw_rewards_dev[idx]
                mb_old_lp = old_log_probs[idx] if old_log_probs is not None else None

                out = get_response_log_probs(policy, mb_ids, mb_lab, return_token_entropy=False)
                log_probs = out["log_probs"]
                token_entropy = None

                loss, meta = grpo_microbatch_train_step(
                    policy_log_probs=log_probs,
                    response_mask=mb_mask,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    loss_type=args.loss_type,
                    raw_rewards=mb_raw,
                    advantages=mb_adv,
                    old_log_probs=mb_old_lp,
                    cliprange=args.cliprange,
                    length_norm=args.length_norm,
                )
                total_loss += loss.item()
                if token_entropy is not None:
                    total_entropy += masked_mean(token_entropy, mb_mask).item()
                if "clip_fraction" in meta:
                    clip_fractions.append(masked_mean(meta["clip_fraction"], mb_mask).item())
                n_micro += 1

                if n_micro % args.gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step_metrics_grad_norm = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)

        # Log step metrics
        step_metrics = {
            "step": step,
            "loss": total_loss / n_micro,
            "entropy": total_entropy / n_micro,
            "rollout_time_s": rollout_time,
            "step_time_s": time.time() - t_step,
            **reward_meta,
        }
        try:
            step_metrics["grad_norm"] = step_metrics_grad_norm
            step_metrics["lr"] = scheduler.get_last_lr()[0]
        except NameError:
            pass
        if clip_fractions:
            step_metrics["clip_fraction"] = sum(clip_fractions) / len(clip_fractions)
        print(f"[step {step}] reward={reward_meta['mean_reward']:.3f} "
              f"loss={step_metrics['loss']:.4f} entropy={step_metrics['entropy']:.3f}")
        log({"type": "train", **step_metrics})

        # Validation
        if step % args.eval_every == 0 or step == args.n_grpo_steps:
            print(f"[step {step}] running validation...")
            load_policy_into_vllm(policy, llm)
            val_reward, _ = evaluate(
                llm, val_examples, prompt_template, r1_zero_reward_fn,
                eval_sampling, n_val=args.n_val
            )
            print(f"[step {step}] val_reward={val_reward:.3f}")
            log({"type": "eval", "step": step, "val_reward": val_reward})

            # Sample a few rollouts for inspection
            sample_examples = random.sample(val_examples, 3)
            for ex in sample_examples:
                pr = prompt_template.format(question=ex["problem"])
                out = llm.generate([pr], eval_sampling)[0]
                log({
                    "type": "sample_rollout",
                    "step": step,
                    "problem": ex["problem"],
                    "ground_truth": ex["answer"],
                    "response": out.outputs[0].text,
                })

    # Save final policy
    final_dir = outdir / "final_policy"
    final_dir.mkdir(exist_ok=True)
    policy.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    metrics_file.close()
    print(f"Saved final policy to {final_dir}")


if __name__ == "__main__":
    main()