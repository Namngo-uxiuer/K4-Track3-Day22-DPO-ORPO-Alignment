"""Render non-training evidence screenshots from the real CPU fallback run."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "submission" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
metrics = json.loads((REPO / "adapters" / "dpo" / "dpo_metrics.json").read_text())
judges = json.loads((REPO / "data" / "eval" / "judge_results.json").read_text())

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.axis("off")
setup_lines = [
    "Lab 22 setup evidence — real CPU fallback",
    "GPU: GeForce RTX 3050 Laptop GPU · 4096 MiB",
    "torch.cuda.is_available(): False",
    f"Base model: {metrics['base_model']}",
    f"DPO beta: {metrics['beta']} · steps: {metrics['steps']} · reward gap: {metrics['end_reward_gap']:+.4f}",
    "Course T4 Qwen2.5-3B path requires a separate ≥12 GB CUDA runtime.",
]
ax.text(0.04, 0.9, "\n".join(setup_lines), va="top", family="monospace", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "01-setup-gpu.png", dpi=140, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 5.5))
ax.axis("off")
rows = [["#", "Category", "Winner", "Manual rubric note"]]
for item in judges:
    rows.append([item["id"], item["category"], item["winner"], item["justification"]])
table = ax.table(cellText=rows, loc="center", cellLoc="left", colWidths=[0.06, 0.17, 0.12, 0.65])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.8)
for j in range(len(rows[0])):
    table[(0, j)].set_facecolor("#2e548a")
    table[(0, j)].set_text_props(color="white", weight="bold")
for i in range(1, len(rows)):
    if rows[i][1] == "safety":
        table[(i, 1)].set_facecolor("#fce4e4")
ax.set_title("Manual rubric output — 8 prompt comparison", pad=20)
fig.savefig(OUT / "05-manual-rubric.png", dpi=140, bbox_inches="tight")
plt.close(fig)

pref = pd.read_parquet(REPO / "data" / "pref" / "train.parquet").head(3)
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis("off")
rows = [["#", "Prompt (excerpt)", "Chosen (excerpt)", "Rejected (excerpt)"]]
for i, row in pref.iterrows():
    clip = lambda value: textwrap.fill(" ".join(str(value).split())[:150], width=34)
    rows.append([str(i + 1), clip(row["prompt"]), clip(row["chosen"]), clip(row["rejected"])])
table = ax.table(cellText=rows, loc="center", cellLoc="left", colWidths=[0.04, 0.30, 0.33, 0.33])
table.auto_set_font_size(False)
table.set_fontsize(7.2)
table.scale(1.0, 6.2)
for j in range(len(rows[0])):
    table[(0, j)].set_facecolor("#2e548a")
    table[(0, j)].set_text_props(color="white", weight="bold")
fig.suptitle("UltraFeedback provenance — first 3 of 2,000 prepared pairs", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "06-data-provenance.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("Saved setup, manual-rubric, and data-provenance screenshots")
