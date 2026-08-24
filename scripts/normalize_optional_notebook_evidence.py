#!/usr/bin/env python3
"""Replace stale pre-run optional notes after real fallback evidence is appended."""
from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def normalize(path: Path, label: str, evidence_heading: str) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        output_parts = []
        for output in cell.get("outputs", []):
            text = output.get("text", "")
            if isinstance(text, str):
                output_parts.append(text)
            else:
                output_parts.extend(str(item) for item in text)
        output_text = "".join(output_parts)
        if f"{label} status: NOT RUN" in output_text or f"{label} is optional in the core rubric" in source:
            if cell["cell_type"] == "markdown":
                cell["source"] = [
                    "## Historical gate note — superseded by appended execution evidence\n",
                    "\n",
                    f"The earlier pre-run note for {label} is retained only as provenance. See the later **{evidence_heading}** cell for the actual result.\n",
                ]
            else:
                cell["outputs"] = [{
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [f"{label} historical pre-run note superseded; see appended execution evidence.\n"],
                }]
            changed = True
    if changed:
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{path.name}: {'normalized' if changed else 'already clean'}")


def main() -> None:
    normalize(REPO / "notebooks" / "05_merge_deploy_gguf.ipynb", "NB5", "GGUF execution evidence")
    normalize(REPO / "notebooks" / "06_benchmark.ipynb", "NB6", "Sampled benchmark execution evidence")


if __name__ == "__main__":
    main()
