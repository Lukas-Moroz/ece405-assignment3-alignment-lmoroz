"""
SFT helper functions for ECE405 Assignment 3.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedTokenizerBase

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Returns:
        dict with keys:
            "input_ids":     (batch_size, max_len - 1)
            "labels":        (batch_size, max_len - 1)  — shifted by 1
            "response_mask": (batch_size, max_len - 1)  — True on response tokens
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    all_full_ids: list[list[int]] = []
    response_starts: list[int] = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        output_ids = tokenizer.encode(output, add_special_tokens=False)
        all_full_ids.append(prompt_ids + output_ids)
        # In labels (shifted by 1), first response token sits at index len(prompt_ids)-1
        response_starts.append(len(prompt_ids) - 1)

    max_len = max(len(ids) for ids in all_full_ids)

    padded_input_ids, padded_labels, padded_masks = [], [], []
    for full_ids, resp_start in zip(all_full_ids, response_starts):
        seq_len = len(full_ids)
        pad_len = max_len - seq_len
        padded = full_ids + [pad_id] * pad_len

        input_ids = padded[:-1]
        labels    = padded[1:]

        n_response = seq_len - 1 - resp_start
        n_padding  = pad_len
        mask = [False] * resp_start + [True] * n_response + [False] * n_padding

        padded_input_ids.append(input_ids)
        padded_labels.append(labels)
        padded_masks.append(mask)

    return {
        "input_ids":     torch.tensor(padded_input_ids, dtype=torch.long),
        "labels":        torch.tensor(padded_labels,    dtype=torch.long),
        "response_mask": torch.tensor(padded_masks,     dtype=torch.bool),
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-token entropy of the next-token prediction distribution.

    Args:
        logits: (batch_size, sequence_length, vocab_size)

    Returns:
        (batch_size, sequence_length)  — H(p) for each position
    """
    # log_softmax is numerically stable (avoids overflow from raw exp)
    log_probs = F.log_softmax(logits, dim=-1)          # (B, T, V)
    probs = torch.exp(log_probs)                        # (B, T, V)
    # H(p) = -sum_v p(v) * log p(v)
    entropy = -(probs * log_probs).sum(dim=-1)          # (B, T)
    return entropy


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probs of the response given the prompt.

    Args:
        model:                HuggingFace causal LM
        input_ids:            (batch_size, seq_len)
        labels:               (batch_size, seq_len) — shifted input_ids
        return_token_entropy: if True, also return per-token entropy

    Returns:
        dict with:
            "log_probs":     (batch_size, seq_len)
            "token_entropy": (batch_size, seq_len)  — only if return_token_entropy=True
    """
    logits = model(input_ids).logits                    # (B, T, V)
    log_probs_all = F.log_softmax(logits, dim=-1)       # (B, T, V)

    # Gather log-prob of each actual next token (the label)
    log_probs = log_probs_all.gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)                                       # (B, T)

    result: dict[str, torch.Tensor] = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float = 1.0,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum tensor elements (where mask=1) along a dimension, then divide by a constant.

    Args:
        tensor:             tensor to reduce
        mask:               same shape as tensor; 1 = include, 0 = exclude
        normalize_constant: divisor after summing
        dim:                dimension to reduce; None = reduce all dimensions

    Returns:
        Normalized sum (masked elements don't contribute).
    """
    masked = tensor * mask  # zero out excluded positions
    if dim is None:
        return masked.sum() / normalize_constant
    return masked.sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute SFT loss for a microbatch and backprop its gradients.

    Loss = -sum(log_probs * response_mask) / normalize_constant
    The loss is divided by gradient_accumulation_steps before .backward()
    so that accumulated gradients are effectively averaged.

    Args:
        policy_log_probs:           (batch_size, seq_len)
        response_mask:              (batch_size, seq_len) — True on response tokens
        gradient_accumulation_steps: number of microbatches per optimizer step
        normalize_constant:         denominator for the masked sum

    Returns:
        (loss, metadata_dict)
    """
    # Normalize by (G × batch_size × normalize_constant) so:
    #   - summed gradients across G microbatches equal the full-batch gradient
    #   - loss represents the mean NLL per token across the effective batch
    batch_size = policy_log_probs.shape[0]
    loss = -masked_normalize(
        policy_log_probs,
        response_mask,
        normalize_constant=normalize_constant,
    ) / (gradient_accumulation_steps * batch_size)
    loss.backward()
    return loss.detach(), {}