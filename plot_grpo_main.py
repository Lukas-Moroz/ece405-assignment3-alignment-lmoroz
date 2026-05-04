import json
from pathlib import Path
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent
PLOTS = REPO / "writeup" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

steps, val = [], []
with open(REPO / "results" / "grpo" / "metrics.jsonl") as f:
    for line in f:
        d = json.loads(line)
        if d.get("type") == "eval":
            steps.append(d["step"])
            val.append(d["val_reward"])

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(steps, val, marker="o", markersize=4, linewidth=2, color="#1f77b4")
ax.axhline(y=0.028, color="gray", linestyle="--", alpha=0.5, label="Zero-shot baseline (2.8%)")
ax.set_xlabel("GRPO step")
ax.set_ylabel("Validation reward (accuracy)")
ax.set_title("§7 GRPO training: validation reward over 200 steps")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
out = PLOTS / "grpo_main.png"
fig.savefig(out, dpi=120)
print(f"Saved {out}")
