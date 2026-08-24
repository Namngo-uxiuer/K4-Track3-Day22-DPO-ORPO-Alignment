# Lab 22 DPO adapter — Ngô Văn Nam

- **MSSV:** 2A202601340
- **Lab:** K4 DPO/ORPO Alignment
- **Base model used for the included local run:** `Qwen/Qwen3.5-0.8B`
- **Adapter method:** LoRA, `r=16`, `lora_alpha=32`
- **Runtime:** CPU fallback on an RTX 3050 Laptop with 4 GB VRAM; PyTorch CUDA was unavailable

## Intended use

This adapter is a teaching artifact for inspecting preference-data preparation,
DPO reward trajectories, qualitative comparison, and GGUF export. It should not
be treated as a production safety model or as a reproduction of the course's
Qwen2.5-3B CUDA/T4 recipe.

## Evidence

The exact preference artifact is in `data/pref/train.parquet` (2,000 rows),
the DPO metrics are in `adapters/dpo/dpo_metrics.json`, and the final reward
gap is shown in `submission/screenshots/03-dpo-reward-curves.png`. The local
GGUF smoke result is in `data/eval/gguf_smoke.json`.

## Hub status

The repository contains this model card, but no Hugging Face Hub push was
performed because no HF token was available in the execution environment.
The status is recorded in `data/eval/external_bonus_status.json`.
