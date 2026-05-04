import json
from pathlib import Path
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent
PLOTS = REPO / "writeup" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 5))
for tag, label in [("ei_G4", "G=4 (batch 512)"), ("ei_G8", "G=8 (batch 1024)")]:
    path = REPO / "results" / "expert_iter" / f"{tag}.json"
    d = json.loads(path.read_text())
    history = d.get("history", [])
    steps = [h["ei_step"] for h in history if "val_accuracy" in h]
    accs = [h["val_accuracy"] for h in history if "val_accuracy" in h]
    ax.plot(steps, accs, marker="o", markersize=8, label=label, linewidth=2)

ax.set_xlabel("EI step")
ax.set_ylabel("Validation accuracy")
ax.set_title("§5 Expert Iteration trajectory")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
out = PLOTS / "ei_trajectory.png"
fig.savefig(out, dpi=120)
print(f"Saved {out}")
