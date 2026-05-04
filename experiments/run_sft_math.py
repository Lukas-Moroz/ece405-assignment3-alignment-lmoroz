"""
SFT on MATH reasoning traces, dataset size sweep + filtered variant.

Usage:
  # Full sweep (128, 256, 512, 1024, full):
  python -u experiments/run_sft_math.py --mode sweep \
      --sft-data data/math/sft.jsonl --val-data data/math/test.jsonl \
      --n-steps 400 --eval-every 80 --n-val-examples 200

  # Filtered only:
  python -u experiments/run_sft_math.py --mode filtered \
      --sft-data data/math/sft.jsonl --val-data data/math/test.jsonl

  # Single size:
  python -u experiments/run_sft_math.py --mode single --dataset-size 128 ...
"""
import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from cs336_alignment.sft import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

REPO = Path(__file__).resolve().parent.parent
DATASET_SIZES = [128, 256, 512, 1024, None]   # None = full
R1_ZERO_PROMPT_PATH = REPO / "cs336_alignment" / "prompts" / "r1_zero.prompt"


def log(msg):
    """Print with timestamp and flush immediately so SLURM logs see it."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_jsonl(path, limit=None):
    data = [json.loads(l) for l in open(path) if l.strip()]
    return data[:limit] if limit else data


def normalize_response_format(response: str) -> str:
    """Insert the space r1_zero_reward_fn expects between </think> and <answer>."""
    return response.replace("</think><answer>", "</think> <answer>")


@torch.no_grad()
def eval_validation(
    model,
    tokenizer,
    val_data,
    prompt_template,
    device,
    max_examples=200,
    max_new_tokens=1024,
):
    """HF-native validation: generation + r1_zero_reward_fn grading."""
    model.eval()
    subset = val_data[:max_examples]
    stop_str = "</answer>"
    total_format = 0.0
    total_answer = 0.0

    for ex in subset:
        prompt = prompt_template.format(question=ex["problem"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        if stop_str in generated:
            generated = generated[: generated.index(stop_str) + len(stop_str)]
        r = r1_zero_reward_fn(generated, ex["answer"])
        total_format += r["format_reward"]
        total_answer += r["answer_reward"]

    n = len(subset)
    model.train()
    return {
        "val_accuracy": total_answer / n,
        "val_format": total_format / n,
        "n_val": n,
    }


def train_one(
    sft_data,
    val_data,
    prompt_template,
    model_path,
    dataset_size,
    n_steps,
    lr,
    grad_accum,
    eval_every,
    n_val_examples,
    tag,
    use_wandb,
    device,
):
    log(f"=== train_one START tag={tag} dataset_size={dataset_size} ===")
    data = sft_data[:dataset_size] if dataset_size else sft_data
    log(f"[{tag}] dataset={len(data)}  steps={n_steps}  lr={lr}  "
        f"eval_every={eval_every}  n_val={n_val_examples}")

    run = None
    if use_wandb:
        import wandb
        run = wandb.init(
            project="ece405-sft-math",
            name=tag,
            config={
                "size": dataset_size, "steps": n_steps, "lr": lr,
                "grad_accum": grad_accum, "eval_every": eval_every,
            },
            reinit=True,
        )

    log(f"[{tag}] loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    log(f"[{tag}] tokenizer loaded")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log(f"[{tag}] loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True
    )
    log(f"[{tag}] model loaded into CPU memory, moving to {device}")
    model = model.to(device)
    log(f"[{tag}] model on device, GPU memory: "
        f"{torch.cuda.memory_allocated(device)/1e9:.1f}GB allocated")
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(n_steps * 0.03)),
        num_training_steps=n_steps,
    )
    optimizer.zero_grad()
    log(f"[{tag}] optimizer + scheduler set up, starting training")

    losses = []
    val_history = []
    t0 = time.time()

    for step in range(1, n_steps + 1):
        step_losses = []
        for _ in range(grad_accum):
            ex = random.choice(data)
            prompt = ex.get("prompt", "")
            response = ex.get("response", ex.get("solution", ""))
            response = normalize_response_format(response)

            tok = tokenize_prompt_and_output([prompt], [response], tokenizer)
            iids = tok["input_ids"].to(device)
            labs = tok["labels"].to(device)
            mask = tok["response_mask"].to(device)

            lp = get_response_log_probs(model, iids, labs)["log_probs"]
            n_resp = mask.sum().clamp(min=1).item()
            loss, _ = sft_microbatch_train_step(
                policy_log_probs=lp,
                response_mask=mask,
                gradient_accumulation_steps=grad_accum,
                normalize_constant=n_resp,
            )
            step_losses.append(loss.item() * grad_accum)

        losses.append(sum(step_losses) / len(step_losses))

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % 20 == 0:
            avg = sum(losses[-20:]) / 20
            log(f"  [{tag}] step={step}/{n_steps}  loss={avg:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s")
            if run:
                run.log({"train/loss": avg, "step": step,
                         "train/lr": scheduler.get_last_lr()[0]})

        if step % eval_every == 0 or step == n_steps:
            t_eval = time.time()
            log(f"  [{tag}] step={step} starting validation eval")
            val_metrics = eval_validation(
                model, tokenizer, val_data, prompt_template,
                device, max_examples=n_val_examples,
            )
            eval_s = time.time() - t_eval
            log(f"  [{tag}] step={step}  VAL accuracy={val_metrics['val_accuracy']:.4f}  "
                f"format={val_metrics['val_format']:.4f}  ({eval_s:.0f}s)")
            val_history.append({"step": step, **val_metrics})
            if run:
                run.log({
                    "val/accuracy": val_metrics["val_accuracy"],
                    "val/format":   val_metrics["val_format"],
                    "step": step,
                })

    ckpt = REPO / "checkpoints" / "sft_math" / tag
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt))
    tokenizer.save_pretrained(str(ckpt))

    final_loss = sum(losses[-min(20, len(losses)):]) / min(20, len(losses))
    final_val = val_history[-1] if val_history else None
    log(f"  [{tag}] final_loss={final_loss:.4f}  "
        f"final_val={final_val}  checkpoint={ckpt}")

    result = {
        "tag": tag,
        "dataset_size": dataset_size or "full",
        "n_steps": n_steps,
        "final_loss": final_loss,
        "losses": losses,
        "val_history": val_history,
    }
    out_dir = REPO / "results" / "sft_math"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{tag}.json", "w") as f:
        json.dump(result, f, indent=2)

    if run:
        run.finish()

    # Free GPU memory before next config
    del model
    del optimizer
    del scheduler
    torch.cuda.empty_cache()
    log(f"=== train_one END tag={tag} ===")

    return final_loss, val_history


def filter_by_reward(sft_data):
    """Filter SFT examples to those graded correct by r1_zero_reward_fn."""
    filtered = []
    for ex in sft_data:
        gt = ex.get("answer", "")
        resp = ex.get("response", "")
        if not gt or not resp:
            continue
        fixed_resp = normalize_response_format(resp)
        r = r1_zero_reward_fn(fixed_resp, gt)
        if r["answer_reward"] > 0.5:
            filtered.append(ex)
    return filtered


def main():
    log("entered main()")
    log(f"python={sys.version_info.major}.{sys.version_info.minor}  "
        f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"cuda_device={torch.cuda.get_device_name(0)}  "
            f"memory={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["sweep", "filtered", "single"], default="sweep")
    p.add_argument("--sft-data", required=True)
    p.add_argument("--val-data", required=True)
    p.add_argument("--model-path", default=str(REPO / "models" / "Qwen2.5-Math-1.5B"))
    p.add_argument("--dataset-size", type=int, default=128)
    p.add_argument("--n-steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=80)
    p.add_argument("--n-val-examples", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    log(f"args: mode={args.mode}  n_steps={args.n_steps}  lr={args.lr}  "
        f"grad_accum={args.grad_accum}  eval_every={args.eval_every}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    log(f"loading sft_data from {args.sft_data}")
    sft_data = load_jsonl(args.sft_data)
    log(f"loaded {len(sft_data)} sft examples")
    log(f"loading val_data from {args.val_data}")
    val_data = load_jsonl(args.val_data)
    log(f"loaded {len(val_data)} val examples")
    prompt_template = R1_ZERO_PROMPT_PATH.read_text()
    log(f"prompt_template loaded ({len(prompt_template)} chars)")

    kw = dict(
        val_data=val_data,
        prompt_template=prompt_template,
        model_path=args.model_path,
        n_steps=args.n_steps,
        lr=args.lr,
        grad_accum=args.grad_accum,
        eval_every=args.eval_every,
        n_val_examples=args.n_val_examples,
        use_wandb=args.use_wandb,
        device=args.device,
    )

    if args.mode == "sweep":
        for size in DATASET_SIZES:
            tag = f"sft_size_{size or 'full'}"
            try:
                train_one(sft_data, dataset_size=size, tag=tag, **kw)
            except Exception as e:
                log(f"!!! train_one for {tag} FAILED: {e}")
                traceback.print_exc()
                log(f"!!! continuing to next config")

    elif args.mode == "filtered":
        log("filtering by reward...")
        filtered = filter_by_reward(sft_data)
        log(f"filtered: {len(sft_data)} → {len(filtered)} correct examples")
        fp = Path(args.sft_data).parent / "sft_filtered.jsonl"
        with open(fp, "w") as f:
            for ex in filtered:
                f.write(json.dumps(ex) + "\n")
        log(f"filtered dataset written to {fp}")
        try:
            train_one(filtered, dataset_size=None, tag="sft_filtered", **kw)
        except Exception as e:
            log(f"!!! train_one for sft_filtered FAILED: {e}")
            traceback.print_exc()

    elif args.mode == "single":
        tag = f"sft_size_{args.dataset_size}"
        train_one(sft_data, dataset_size=args.dataset_size, tag=tag, **kw)

    log("main() finished")


if __name__ == "__main__":
    main()