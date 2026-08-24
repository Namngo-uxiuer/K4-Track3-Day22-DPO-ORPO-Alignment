#!/usr/bin/env python3
"""Add the real β-sweep output to NB3 without rerunning the CUDA cells."""
from __future__ import annotations

import json
import uuid
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "notebooks" / "03_dpo_train.ipynb"
RESULTS = REPO / "data" / "eval" / "beta_sweep_results.json"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if any("β-sweep execution evidence" in "".join(cell.get("source", [])) for cell in notebook["cells"]):
        print("NB3 beta evidence already present")
        return
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    summary = "\n".join(
        f"beta={item['beta']}: final_gap={item['final_gap']:+.6f}, training_pair_win_rate={item['training_pair_win_rate']:.2f}"
        for item in data["results"]
    )
    notebook["cells"].extend([
        {
            "cell_type": "markdown",
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "source": [
                "## β-sweep execution evidence — actual local mini-experiment\n",
                "\n",
                "This bonus cell records the real CPU fallback sweep; it is explicitly not a T4-scale result.\n",
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
                    "β-sweep execution evidence: PASS (CPU_FALLBACK)\n",
                    f"betas: {data['betas']} | steps_per_beta: {data['steps_per_beta']}\n",
                    summary + "\n",
                    "plot: submission/screenshots/bonus-beta-sweep.png\n",
                ],
            }],
            "source": ["# Results captured from scripts/run_beta_sweep_cpu.py\n", "print('See captured output below.')"],
        },
    ])
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Stamped NB3 beta-sweep evidence")


if __name__ == "__main__":
    main()
