"""Plot §8 ablation results."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent
PLOTS_DIR = REPO / "writeup" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_curve(path):
    steps, val = [], []
    if not Path(path).exists():
        return steps, val
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("type") == "eval":
                steps.append(d["step"])
                val.append(d["val_reward"])
    return steps, val


def plot(curves, title, savename):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, path in curves:
        steps, val = load_curve(path)
        if not steps:
            print(f"WARN: {path} missing or empty, skipping {label}")
            continue
        ax.plot(steps, val, label=label, marker="o", markersize=4)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("Validation reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = PLOTS_DIR / savename
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


plot([
    ("reinforce_with_baseline", "results/grpo/metrics.jsonl"),
    ("no_baseline",             "results/grpo_ablations/no_baseline/metrics.jsonl"),
], "§8.2 Baselines", "ablation_baselines.png")

plot([
    ("std norm OFF (Dr. GRPO)", "results/grpo/metrics.jsonl"),
    ("std norm ON (DeepSeek)",  "results/grpo_ablations/std_norm_on/metrics.jsonl"),
], "§8.4 Group standard deviation", "ablation_std.png")

plot([
    ("r1_zero",       "results/grpo/metrics.jsonl"),
    ("question_only", "results/grpo_ablations/question_only/metrics.jsonl"),
], "§8.7 Prompt", "ablation_prompt.png")

plot([
    ("masked_mean",      "results/grpo/metrics.jsonl"),
    ("masked_normalize", "results/grpo_ablations/length_normalize/metrics.jsonl"),
], "§8.3 Length normalization", "ablation_length_norm.png")

print("Done.")
