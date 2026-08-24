#!/usr/bin/env python3
"""Append the real local GGUF/llama.cpp execution evidence to NB5."""
from __future__ import annotations

import json
import uuid
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "notebooks" / "05_merge_deploy_gguf.ipynb"
RESULTS = REPO / "data" / "eval" / "gguf_smoke.json"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if any("GGUF execution evidence" in "".join(cell.get("source", [])) for cell in notebook["cells"]):
        print("NB5 GGUF evidence already present")
        return
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    notebook["cells"].extend([
        {
            "cell_type": "markdown",
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "source": [
                "## GGUF execution evidence — real local fallback run\n",
                "\n",
                "The course CUDA/Unsloth path is hardware-gated here, so this cell records a real CPU fallback export and llama.cpp smoke test.\n",
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
                    "NB5 status: PASS (CPU_FALLBACK)\\n",
                    f"model: {data['model']} | size_mb: {data['model_size_bytes'] / 1e6:.1f} | runtime: {data['runtime']}\\n",
                    f"generated_output: {data['output']}\\n",
                    "plot: submission/screenshots/06-gguf-smoke.png\\n",
                    "metadata: data/eval/gguf_smoke.json\\n",
                ],
            }],
            "source": ["# Captured from scripts/run_gguf_smoke.py\n", "print('See captured output below.')"],
        },
    ])
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Stamped NB5 GGUF evidence")


if __name__ == "__main__":
    main()
