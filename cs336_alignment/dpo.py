"""
DPO (Direct Preference Optimization) loss implementation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def _sum_response_log_probs(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    response: str,
) -> Tensor:
    """Sum of log-probs over response tokens conditioned on prompt.

    Tokenizes prompt+response as a single string. The response boundary is
    identified by tokenizing the prompt alone (no special tokens) and counting
    the resulting tokens.
    """
    # Tokenize prompt separately (no special tokens) to find boundary
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    n_prompt = len(prompt_ids)

    # Tokenize full text as one string (no special tokens)
    full_ids = tokenizer.encode(prompt + response, add_special_tokens=False)

    input_tensor = torch.tensor([full_ids], dtype=torch.long)
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    logits = model(input_tensor).logits                   # (1, T, V)
    log_probs_all = F.log_softmax(logits, dim=-1)         # (1, T, V)

    # log_probs_all[:, t, :] predicts token at position t+1
    # → gather the log-prob of each actual next token
    token_log_probs = log_probs_all[:, :-1, :].gather(
        dim=-1, index=input_tensor[:, 1:].unsqueeze(-1)
    ).squeeze(-1)  # (1, T-1)

    # Response starts at position n_prompt in the full_ids.
    # In token_log_probs, log p(full_ids[t] | ...) is at index t-1.
    # So response log probs are at indices [n_prompt-1, n_prompt, ..., T-2].
    response_log_probs = token_log_probs[:, n_prompt - 1:]  # (1, n_response)
    return response_log_probs.sum()


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> Tensor:
    """DPO loss for a single preference pair.

    L_DPO = -log σ(β * (log π(y_w|x) - log π_ref(y_w|x))
                      - β * (log π(y_l|x) - log π_ref(y_l|x)))

    Args:
        lm:               LM being trained
        lm_ref:           frozen reference LM
        tokenizer:        shared tokenizer
        beta:             KL regularisation strength
        prompt:           conditioning context
        response_chosen:  preferred response y_w
        response_rejected: rejected response y_l

    Returns:
        scalar DPO loss
    """
    # Log-probs under the policy being trained
    lm_lp_chosen   = _sum_response_log_probs(lm,     tokenizer, prompt, response_chosen)
    lm_lp_rejected = _sum_response_log_probs(lm,     tokenizer, prompt, response_rejected)

    # Log-probs under the frozen reference model (no grad)
    with torch.no_grad():
        ref_lp_chosen   = _sum_response_log_probs(lm_ref, tokenizer, prompt, response_chosen)
        ref_lp_rejected = _sum_response_log_probs(lm_ref, tokenizer, prompt, response_rejected)

    # DPO implicit reward difference
    chosen_reward   = beta * (lm_lp_chosen   - ref_lp_chosen)
    rejected_reward = beta * (lm_lp_rejected - ref_lp_rejected)

    loss = -F.logsigmoid(chosen_reward - rejected_reward)
    return loss
