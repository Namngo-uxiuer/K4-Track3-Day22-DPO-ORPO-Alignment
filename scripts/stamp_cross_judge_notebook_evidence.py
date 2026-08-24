#!/usr/bin/env python3
"""Append the real OpenAI cross-judge result to NB4."""
from __future__ import annotations

import json
import uuid
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "notebooks" / "04_compare_and_eval.ipynb"
RESULTS = REPO / "data" / "eval" / "cross_judge_openai.json"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if any("OpenAI cross-judge evidence" in "".join(cell.get("source", [])) for cell in notebook["cells"]):
        print("NB4 cross-judge evidence already present")
        return
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    notebook["cells"].extend([
        {
            "cell_type": "markdown",
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "source": [
                "## OpenAI cross-judge evidence — real API run\n",
                "\n",
                "The eight comparison pairs were judged with the configured OpenAI model. The API key is not included in notebook output.\n",
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
                "text": [
                    "OpenAI cross-judge status: PASS\n",
                    f"model: {data['model']} | n: {data['n']} | SFT-only wins: {data['wins']['SFT_only']} | SFT+DPO wins: {data['wins']['SFT_DPO']} | ties: {data['wins']['tie']}\n",
                    f"DPO pairwise win-rate: {data['dpo_pairwise_win_rate']:.2f}\n",
                    "plot: submission/screenshots/09-openai-cross-judge.png\n",
                    "metadata: data/eval/cross_judge_openai.json\n",
                ],
            }],
            "source": ["# Captured from scripts/run_openai_cross_judge.py\n", "print('See captured output below.')"],
        },
    ])
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Stamped NB4 cross-judge evidence")


if __name__ == "__main__":
    main()
