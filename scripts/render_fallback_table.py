"""Render the saved CPU fallback comparison as a readable wrapped table PNG."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parent.parent
records = [json.loads(line) for line in (REPO / "data" / "eval" / "side_by_side.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
rows = [["#", "Category", "Prompt", "SFT-only", "SFT+DPO"]]
for r in records:
    rows.append([
        r["id"],
        r["category"],
        textwrap.fill(r["prompt"], 34),
        textwrap.fill(r["sft_only"], 44),
        textwrap.fill(r["sft_dpo"], 44),
    ])

fig, ax = plt.subplots(figsize=(20, 12.5))
ax.axis("off")
table = ax.table(
    cellText=rows,
    loc="center",
    cellLoc="left",
    colWidths=[0.04, 0.10, 0.20, 0.33, 0.33],
)
table.auto_set_font_size(False)
table.set_fontsize(7.5)
table.scale(1.0, 3.7)
for j in range(len(rows[0])):
    table[(0, j)].set_facecolor("#2e548a")
    table[(0, j)].set_text_props(color="white", weight="bold")
for i in range(1, len(rows)):
    if rows[i][1] == "safety":
        table[(i, 1)].set_facecolor("#fce4e4")
fig.savefig(REPO / "submission" / "screenshots" / "04-side-by-side-table.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("Saved readable wrapped side-by-side table")
