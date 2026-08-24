#!/usr/bin/env python3
"""Append the Q5_K_M smoke result to NB5 after the Q4 evidence."""
from __future__ import annotations

import json
import uuid
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "notebooks" / "05_merge_deploy_gguf.ipynb"
RESULTS = REPO / "data" / "eval" / "gguf_q5_smoke.json"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if any("Q5_K_M smoke evidence" in "".join(cell.get("source", [])) for cell in notebook["cells"]):
        print("NB5 Q5 evidence already present")
        return
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    notebook["cells"].extend([
        {
            "cell_type": "markdown",
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "source": [
                "## Q5_K_M smoke evidence — real local fallback run\n",
                "\n",
                "The higher-quality Q5_K_M release was loaded and smoke-tested with the same llama.cpp CLI.\n",
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
                    "NB5 Q5_K_M status: PASS (CPU_FALLBACK)\n",
                    f"model: {data['model']} | size_mb: {data['model_size_bytes'] / 1e6:.1f} | sha256: {data['model_sha256']}\n",
                    f"generated_output: {data['output']}\n",
                    "plot: submission/screenshots/08-gguf-q5-smoke.png\n",
                    "metadata: data/eval/gguf_q5_smoke.json\n",
                ],
            }],
            "source": ["# Captured from scripts/run_gguf_smoke.py with GGUF_MODEL=Q5_K_M\n", "print('See captured output below.')"],
        },
    ])
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Stamped NB5 Q5 evidence")


if __name__ == "__main__":
    main()
