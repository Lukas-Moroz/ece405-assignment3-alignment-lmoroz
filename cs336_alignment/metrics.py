"""
Metrics parsing helpers for MMLU and GSM8K.
"""
from __future__ import annotations

import re
from typing import Any


def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """Parse model output into one of A/B/C/D for MMLU.

    Strategy: find the first standalone letter A–D in the model output.
    Returns None if no such letter is found.
    """
    # Look for an isolated A/B/C/D (word boundary on both sides)
    match = re.search(r'\b([A-D])\b', model_output)
    if match:
        return match.group(1)
    return None


def parse_gsm8k_response(model_output: str) -> str | None:
    """Parse model output into a numeric string by taking the last number.

    Returns None if no number is found in the output.
    """
    # Find all numbers (integers or decimals, optionally with commas)
    numbers = re.findall(r'-?\d[\d,]*(?:\.\d+)?', model_output)
    if numbers:
        # Return the last one found, stripping commas
        return numbers[-1].replace(',', '')
    return None
