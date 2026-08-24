# Submission audit — 2026-08-24

**Student:** Ngô Văn Nam — **MSSV:** 2A202601340

## Core gatekeeper

| Requirement | Status | Evidence |
|---|---|---|
| SFT adapter `r=16`, `lora_alpha=32` | PASS | `adapters/sft-mini/adapter_config.json` |
| DPO adapter distinct from SFT | PASS | `adapters/dpo/adapter_config.json`, separate safetensors |
| SFT loss decreases | PASS | `submission/screenshots/02-sft-loss.png`, `cpu_fallback_training.json` |
| SFT generation | PASS | `data/eval/sft_generation.txt` |
| Preference parquet schema | PASS | `data/pref/train.parquet`: 2,000 rows, `prompt/chosen/rejected` |
| Chosen != rejected samples | PASS | `data/pref/ultrafeedback_metadata.json` |
| DPO reward curves | PASS | `submission/screenshots/03-dpo-reward-curves.png` |
| Reward gap final positive | PASS | final gap `+8.952895` |
| 8 prompts × 2 models | PASS | `data/eval/side_by_side.jsonl`, `04-side-by-side-table.png` |
| Win/loss/tie summary | PASS | `0 / 1 / 7`; `05-manual-rubric.png` |
| Reflection sections | PASS | `submission/REFLECTION.md`, §3 and §6 exceed 150 words |
| Reproducible scripts | PASS | `scripts/run_cpu_fallback.py`, `scripts/prepare_ultrafeedback.py` |

## Optional / hardware-gated

- β-sweep mini: **PASS (CPU fallback)**; see `data/eval/beta_sweep_results.json` and `submission/screenshots/bonus-beta-sweep.png`.
- NB5 GGUF + llama.cpp smoke: **not run**; no fake GGUF or screenshot was created.
- NB6 IFEval/GSM8K/MMLU/AlpacaEval-lite: **not run**; no fake benchmark scores were created.
- External API judge/cross-judge: **not run**; manual rubric is included.

## Important scope note

The local evidence is a real `CPU_FALLBACK` run on Qwen3.5-0.8B. It is not the exact T4 recipe (Qwen2.5-3B-bnb-4bit, CUDA, 1k Vietnamese Alpaca SFT, and 2k UltraFeedback DPO). The exact UltraFeedback data-prep artifact is present and independently validated; the included adapter metrics disclose the smaller fallback training sample.
