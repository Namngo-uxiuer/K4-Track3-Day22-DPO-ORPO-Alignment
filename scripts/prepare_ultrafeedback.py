#!/usr/bin/env python3
"""Prepare the lab's exact UltraFeedback slice without requiring a dataset builder.

The local Hugging Face cache already contains the Arrow shard.  Reading it
directly avoids a stale Windows file lock left by an interrupted dataset
builder, while preserving the original dataset provenance.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "pref"
ARROW = Path(os.environ.get(
    "ULTRA_FEEDBACK_ARROW",
    r"C:\Users\Acer\.cache\huggingface\datasets\argilla___ultrafeedback-binarized-preferences-cleaned\default\0.0.0\770076f077c4c5e298498fa32f804857f46d5134\ultrafeedback-binarized-preferences-cleaned-train.arrow",
))
TRAIN_ROWS = 2000
EVAL_ROWS = 200


def assistant_text(messages) -> str:
    if isinstance(messages, list):
        assistants = [m.get("content", "") for m in messages if m.get("role") == "assistant"]
        if assistants:
            return str(assistants[-1]).strip()
    return str(messages).strip()


def main() -> None:
    if not ARROW.exists():
        raise FileNotFoundError(f"UltraFeedback Arrow cache not found: {ARROW}")
    dataset = Dataset.from_file(str(ARROW))
    selected = dataset.select(range(min(TRAIN_ROWS + EVAL_ROWS, len(dataset))))
    rows = []
    for item in selected:
        prompt = str(item["prompt"]).strip()
        chosen = assistant_text(item["chosen"])
        rejected = assistant_text(item["rejected"])
        if prompt and chosen and rejected and chosen != rejected:
            rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    if len(rows) < TRAIN_ROWS + EVAL_ROWS:
        raise ValueError(f"Expected {TRAIN_ROWS + EVAL_ROWS} valid rows, found {len(rows)}")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows[:TRAIN_ROWS]).to_parquet(OUT / "train.parquet", index=False)
    pd.DataFrame(rows[TRAIN_ROWS:TRAIN_ROWS + EVAL_ROWS]).to_parquet(OUT / "eval.parquet", index=False)
    metadata = {
        "dataset": "argilla/ultrafeedback-binarized-preferences-cleaned",
        "source_arrow": str(ARROW),
        "source_rows": len(dataset),
        "train_rows": TRAIN_ROWS,
        "eval_rows": EVAL_ROWS,
        "columns": ["prompt", "chosen", "rejected"],
        "validation": {
            "chosen_rejected_distinct": all(r["chosen"] != r["rejected"] for r in rows),
            "nonempty": all(all(r[c] for c in ("prompt", "chosen", "rejected")) for r in rows),
        },
        "note": "Exact lab data-prep artifact. The local CPU fallback training run uses a small sampled subset because the machine cannot run the T4 recipe.",
    }
    (OUT / "ultrafeedback_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "README.md").write_text(
        "# Preference data\n\n"
        "`train.parquet` and `eval.parquet` are prepared from "
        "`argilla/ultrafeedback-binarized-preferences-cleaned` with the lab schema "
        "`prompt`, `chosen`, `rejected`. See `ultrafeedback_metadata.json` for provenance.\n\n"
        "The adapter included in this repository is explicitly labelled `CPU_FALLBACK`; "
        "the exact T4 recipe was not run on this 4 GB GPU/CPU-only PyTorch environment.\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
