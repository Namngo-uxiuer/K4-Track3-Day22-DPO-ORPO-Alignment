#!/usr/bin/env python3
"""Run the real CPU DPO demonstration on a sample of the exact lab dataset."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import run_cpu_fallback as fallback


REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    frame = pd.read_parquet(REPO / "data" / "pref" / "train.parquet")
    sample_rows = int(fallback.os.environ.get("CPU_DPO_PREF_SAMPLE", "32"))
    rows = frame.head(sample_rows).to_dict("records")
    tokenizer = fallback.load_base(tokenizer_only=True)
    metrics = fallback.run_dpo(rows, tokenizer)
    metrics.update({
        "preference_dataset": "argilla/ultrafeedback-binarized-preferences-cleaned",
        "prepared_train_rows": len(frame),
        "cpu_training_sample_rows": len(rows),
        "data_provenance": "Exact lab dataset prepared in data/pref/train.parquet; CPU demonstration uses the first sampled rows.",
    })
    (REPO / "adapters" / "dpo" / "dpo_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "prepared_train_rows": len(frame),
        "cpu_training_sample_rows": len(rows),
        "reward_gap": metrics["end_reward_gap"],
        "elapsed_seconds": metrics["elapsed_seconds"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
