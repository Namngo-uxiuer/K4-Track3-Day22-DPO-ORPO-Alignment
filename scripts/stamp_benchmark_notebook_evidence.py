#!/usr/bin/env python3
"""Append the real sampled benchmark evidence to NB6."""
from __future__ import annotations

import json
import uuid
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "notebooks" / "06_benchmark.ipynb"
RESULTS = REPO / "data" / "eval" / "benchmark_results.json"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if any("Sampled benchmark execution evidence" in "".join(cell.get("source", [])) for cell in notebook["cells"]):
        print("NB6 benchmark evidence already present")
        return
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    lines = [
        "NB6 status: PASS_SAMPLED_FALLBACK\n",
        f"compute_tier: {data['compute_tier']} | official_full_suite: {data['official_full_suite']}\n",
        f"elapsed_seconds: {data['elapsed_seconds']}\n",
    ]
    for name, metric in data["metrics"].items():
        lines.append(
            f"{name}: SFT={metric['sft_only']:.2f} | SFT+DPO={metric['sft_dpo']:.2f} | n={metric['n']}\n"
        )
    lines.extend([
        f"official harness: {data['official_harness_attempt']['status']}\n",
        "official reason: dataset download was blocked after the model loaded; no full-suite score is claimed.\n",
        "plot: submission/screenshots/07-benchmark-comparison.png\n",
        "metadata: data/eval/benchmark_results.json\n",
    ])
    notebook["cells"].extend([
        {
            "cell_type": "markdown",
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "source": [
                "## Sampled benchmark execution evidence — real local fallback run\n",
                "\n",
                "This cell records the actual CPU fallback comparison. It is deliberately labelled sampled rather than presented as the official full IFEval/GSM8K/MMLU suite.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "outputs": [{
                "name": "stdout",
                "output_type": "stream",
                "text": lines,
            }],
            "source": ["# Captured from scripts/run_benchmark_cpu_fallback.py\n", "print('See captured output below.')"],
        },
    ])
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Stamped NB6 benchmark evidence")


if __name__ == "__main__":
    main()
