"""
Download models needed for ECE405 Assignment 3.

Models downloaded:
  - Qwen/Qwen2.5-Math-1.5B      (main model for reasoning experiments)
  - Qwen/Qwen2.5-0.5B           (tiny baseline)
  - Qwen/Qwen2.5-3B-Instruct    (medium instruct model)

Saved to a local 'models/' directory next to the repo.

Usage:
    uv run --no-sync python download_models.py

    # Download only the math model (fastest, needed first):
    uv run --no-sync python download_models.py --only math
"""
import argparse
import os
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "math":   "Qwen/Qwen2.5-Math-1.5B",       # required for assignment
    "tiny":   "Qwen/Qwen2.5-0.5B",             # optional small baseline
    "medium": "Qwen/Qwen2.5-3B-Instruct",      # optional medium instruct
}

# Save models in a 'models/' folder alongside the repo
SAVE_BASE = Path("models")


def download_model(model_name: str, save_dir: Path):
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"Saving to:   {save_dir}")
    print(f"{'='*60}")
    save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(save_dir)
    print(f"  Tokenizer saved.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        # Use float16 to cut disk/memory usage in half on HPC
        torch_dtype="auto",
    )
    model.save_pretrained(save_dir)
    print(f"  Model saved.")
    print(f"Done: {model_name} -> {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download models for ECE405 A3")
    parser.add_argument(
        "--only",
        choices=list(MODELS.keys()),
        default=None,
        help="Download only one specific model (default: all)",
    )
    parser.add_argument(
        "--save_base",
        type=str,
        default=str(SAVE_BASE),
        help=f"Base directory to save models (default: {SAVE_BASE})",
    )
    args = parser.parse_args()

    save_base = Path(args.save_base)
    to_download = {args.only: MODELS[args.only]} if args.only else MODELS

    print(f"Will download {len(to_download)} model(s) to: {save_base.resolve()}")
    for key, model_name in to_download.items():
        # Save as models/Qwen2.5-Math-1.5B etc.
        model_short_name = model_name.split("/")[-1]
        save_dir = save_base / model_short_name
        download_model(model_name, save_dir)

    print(f"\nAll done! Models saved under: {save_base.resolve()}")
    print("Update --model_path in your eval script to point to:")
    for key, model_name in to_download.items():
        model_short_name = model_name.split("/")[-1]
        print(f"  {save_base / model_short_name}")


if __name__ == "__main__":
    main()
