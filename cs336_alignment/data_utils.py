"""
Dataset utilities: packed SFT dataset and batch iteration.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

_ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{prompt}\n\n"
    "### Response:\n{response}"
)


class PackedSFTDataset(Dataset):
    """Packs tokenized instruction-tuning examples into fixed-length chunks.

    Each example in the source file is a JSON object with "prompt" and
    "response" keys. The Alpaca template is applied before tokenization.
    All token IDs are concatenated and sliced into non-overlapping (input_ids,
    labels) pairs of length `seq_length` using the standard LM next-token
    formulation: labels[i] = input_ids[i+1].
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | Path,
        seq_length: int,
        shuffle: bool,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length

        # 1. Load all documents
        docs: list[dict] = []
        with open(dataset_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                docs.append(json.loads(line))

        if shuffle:
            random.shuffle(docs)

        # 2. Tokenize with Alpaca template and concatenate into one long token stream
        all_ids: list[int] = []
        for doc in docs:
            if "prompt" in doc and "response" in doc:
                text = _ALPACA_TEMPLATE.format(
                    prompt=doc["prompt"], response=doc["response"]
                )
            elif "text" in doc:
                text = doc["text"]
            else:
                text = " ".join(str(v) for v in doc.values())
            ids = tokenizer.encode(text, add_special_tokens=True)
            all_ids.extend(ids)
            # Append EOS between documents so the model sees clean document boundaries
            if tokenizer.eos_token_id is not None:
                all_ids.append(tokenizer.eos_token_id)

        # 3. Pack into non-overlapping windows of seq_length
        #    input_ids[k] = all_ids[k*L : (k+1)*L]
        #    labels[k]    = all_ids[k*L+1 : (k+1)*L+1]
        #    => need all_ids to have at least (k+1)*L + 1 tokens,
        #       i.e. k in range((T-1) // L)
        T = len(all_ids)
        n_examples = (T - 1) // seq_length
        self.examples: list[dict[str, Tensor]] = []
        for i in range(n_examples):
            lo = i * seq_length
            input_ids = torch.tensor(all_ids[lo : lo + seq_length], dtype=torch.long)
            labels    = torch.tensor(all_ids[lo + 1 : lo + seq_length + 1], dtype=torch.long)
            self.examples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return self.examples[idx]


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Return a DataLoader that yields batches of size `batch_size`.

    Iterating through the returned DataLoader constitutes one epoch.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
