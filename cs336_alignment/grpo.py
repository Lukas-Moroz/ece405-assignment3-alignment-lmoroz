"""
GRPO (Group Relative Policy Optimization) helper functions.
"""
from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

# Masked mean
def masked_mean(
    tensor: Tensor,
    mask: Tensor,
    dim: int | None = None,
) -> Tensor:
    """Mean of tensor over positions where mask==1.

    When all mask values are 0 along the reduced dimension, the result
    is NaN (undefined mean), matching the expected snapshot behaviour.
    """
    masked = tensor * mask
    if dim is None:
        return masked.sum() / mask.sum().clamp(min=1.0)
    denom = mask.float().sum(dim=dim)            # (B, ...) counts of True
    result = masked.sum(dim=dim) / denom.clamp(min=1.0)
    # Where no masked elements exist, the mean is undefined → NaN
    result = torch.where(denom > 0, result, torch.full_like(result, float('nan')))
    return result

# Group-normalized rewards (GRPO advantage estimation)
def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Score rollouts and compute group-normalized advantages.

    Responses are grouped in consecutive blocks of `group_size`.  Within each
    group the mean (and optionally std) of the raw rewards are used to
    produce zero-mean (unit-variance) advantages.

    Returns:
        advantages:  (rollout_batch_size,) group-normalised rewards
        raw_rewards: (rollout_batch_size,) un-normalised scalar rewards
        metadata:    dict of summary statistics
    """
    n = len(rollout_responses)
    assert n % group_size == 0, "rollout_batch_size must be divisible by group_size"

    # --- score every response ---
    all_rewards: list[float] = []
    all_format_rewards: list[float] = []
    all_answer_rewards: list[float] = []

    for response, gt in zip(rollout_responses, repeated_ground_truths):
        scores = reward_fn(response, gt)
        all_rewards.append(float(scores["reward"]))
        all_format_rewards.append(float(scores["format_reward"]))
        all_answer_rewards.append(float(scores["answer_reward"]))

    raw_rewards = torch.tensor(all_rewards, dtype=torch.float32)       # (N,)

    #group-normalise
    advantages = torch.zeros_like(raw_rewards)
    n_groups = n // group_size
    for g in range(n_groups):
        lo, hi = g * group_size, (g + 1) * group_size
        group = raw_rewards[lo:hi]
        mean = group.mean()
        if normalize_by_std:
            # unbiased=True (N-1 denominator) matches the reference implementation
            std = group.std(unbiased=True)
            advantages[lo:hi] = (group - mean) / (std + advantage_eps)
        else:
            advantages[lo:hi] = group - mean

    metadata: dict[str, float] = {
        "mean_reward":        raw_rewards.mean().item(),
        "mean_format_reward": sum(all_format_rewards) / len(all_format_rewards),
        "mean_answer_reward": sum(all_answer_rewards) / len(all_answer_rewards),
        "std_reward":         (raw_rewards.std() + 1e-8).item(),
        "frac_correct":       float((raw_rewards > 0).float().mean().item()),
    }
    return advantages.float(), raw_rewards.float(), metadata

# Per-token policy-gradient losses
def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
) -> Tensor:
    """REINFORCE-style per-token loss.

    Loss = -advantage * log_prob  (per token, broadcast advantage over seq dim)

    Args:
        raw_rewards_or_advantages: (batch_size, 1)
        policy_log_probs:          (batch_size, sequence_length)

    Returns:
        (batch_size, sequence_length)
    """
    # advantages shape (B, 1) broadcasts over sequence length
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """PPO-clip loss as used in GRPO.

    ratio  = exp(log π - log π_old)
    L_clip = -min(ratio * A,  clip(ratio, 1-ε, 1+ε) * A)

    Args:
        advantages:      (batch_size, 1)
        policy_log_probs:(batch_size, sequence_length)
        old_log_probs:   (batch_size, sequence_length)
        cliprange:       ε

    Returns:
        per-token loss (batch_size, sequence_length) and metadata dict.
    """
    log_ratio = policy_log_probs - old_log_probs          # (B, T)
    ratio = torch.exp(log_ratio)                           # (B, T)

    # advantages (B, 1) broadcasts to (B, T)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * ratio.clamp(1.0 - cliprange, 1.0 + cliprange)
    loss = torch.max(pg_loss1, pg_loss2)                   # (B, T)

    # fraction of tokens that were clipped
    clipped = (ratio < 1.0 - cliprange) | (ratio > 1.0 + cliprange)
    metadata = {"clip_fraction": clipped.float()}
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: Tensor | None = None,
    advantages: Tensor | None = None,
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Dispatcher for the three policy-gradient loss variants.

    Returns (per-token loss tensor, metadata dict).
    """
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}

    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {}

    elif loss_type == "grpo_clip":
        assert advantages is not None and old_log_probs is not None and cliprange is not None
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)

    elif loss_type == "grpo_no_clip":
        assert advantages is not None and old_log_probs is not None
        return compute_grpo_no_clip_loss(advantages, policy_log_probs, old_log_probs)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")

# GRPO microbatch train step
def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: Tensor | None = None,
    advantages: Tensor | None = None,
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    length_norm: str = "mean",
    normalize_constant: float = 1024.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Forward + backward for one GRPO microbatch.

    Computes the per-token policy-gradient loss, masks to response tokens,
    averages over the effective batch, then calls .backward().

    Returns the scalar loss (before scaling) and metadata.
    """
    # per-token losses, shape (B, T)
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # Average per-token loss over response tokens, then scale for grad accumulation.
    # masked_mean gives the mean per-token PG loss; dividing by G ensures gradients
    # are averaged across accumulation steps.
    if length_norm == "mean":
        loss = masked_mean(per_token_loss, response_mask)
    elif length_norm == "normalize":
        masked = per_token_loss * response_mask
        loss = masked.sum() / (response_mask.shape[0] * normalize_constant)
    else:
        raise ValueError(f"Unknown length_norm: {length_norm!r}")
    if not torch.isfinite(loss):
        print(f'[grpo] non-finite loss detected, skipping step')
        return loss.detach(), metadata
    (loss / gradient_accumulation_steps).backward()
    return loss.detach(), metadata

# GRPO No-Clip loss (off-policy ablation from §8)

def compute_grpo_no_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Unclipped off-policy PG loss: -ratio * A, per token."""
    ratio = torch.exp(policy_log_probs - old_log_probs)
    loss = -advantages * ratio
    return loss, {}
