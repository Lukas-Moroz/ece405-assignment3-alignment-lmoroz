import json
from pathlib import Path
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent
PLOTS = REPO / "writeup" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

# Load val histories from each size run
sizes_runs = [
    ("sft_size_128", 128),
    ("sft_size_256", 256),
    ("sft_size_512", 512),
    ("sft_size_1024", 1024),
    ("sft_size_full", "full (9997)"),
]

fig, ax = plt.subplots(figsize=(8, 5))
for tag, label in sizes_runs:
    path = REPO / "results" / "sft_math" / f"{tag}.json"
    if not path.exists():
        print(f"missing {path}")
        continue
    d = json.loads(path.read_text())
    history = d.get("val_history", [])
    if not history:
        continue
    steps = [h["step"] for h in history]
    accs  = [h["val_accuracy"] for h in history]
    ax.plot(steps, accs, marker="o", markersize=5, label=f"size={label}")

ax.set_xlabel("SFT step")
ax.set_ylabel("Validation accuracy")
ax.set_title("§4 SFT dataset size sweep")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
out = PLOTS / "sft_size_sweep.png"
fig.savefig(out, dpi=120)
print(f"Saved {out}")
