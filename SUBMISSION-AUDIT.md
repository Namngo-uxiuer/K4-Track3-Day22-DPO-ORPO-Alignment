# Submission audit — 2026-08-24

**Student:** Ngô Văn Nam — **MSSV:** 2A202601340
**Public GitHub:** https://github.com/Namngo-uxiuer/K4-Track3-Day22-DPO-ORPO-Alignment
**Verified evidence commit:** `d22a941` (pushed to `origin/main`)

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
- NB5 GGUF Q4_K_M + Q5_K_M + llama.cpp smoke: **PASS (CPU fallback; llama.cpp b10605 CPU CLI; Q4 smoke + Q5 smoke PASS)**; see `data/eval/gguf_smoke.json`, `data/eval/gguf_q5_smoke.json`, and `submission/screenshots/06-gguf-smoke.png`/`08-gguf-q5-smoke.png`.
- NB6 IFEval/GSM8K/MMLU/AlpacaEval-lite: **PASS (sampled CPU fallback + official MMLU subject/GSM8K/IFEval smokes; full suites CPU-time limited)**; see `data/eval/benchmark_results.json` and `submission/screenshots/07-benchmark-comparison.png`.
- Official benchmark smoke evidence: **PASS**; see `data/eval/official_benchmark_smokes.json` and `submission/screenshots/10-official-mmlu-smoke.png`/`11-official-benchmark-smokes.png`.
- External API judge/cross-judge: **PASS (gpt-4o-mini; 3 DPO wins / 5 SFT wins / 0 ties)**; see `data/eval/cross_judge_openai.json` and `submission/screenshots/09-openai-cross-judge.png`.
- Hugging Face Hub push: **PASS (https://huggingface.co/Namdzfsfds/lab22-dpo-vn)**; see `data/eval/hf_push_status.json` and `submission/HF_MODEL_CARD.md`.
- W&B public run: **blocked — no WANDB_API_KEY**; no fabricated public URL.
- Public GGUF binary upload: **blocked — GitHub LFS permission for this public fork**; local Q4/Q5 files, hashes, and reproducible scripts are retained.

## Important scope note

The local evidence is a real `CPU_FALLBACK` run on Qwen3.5-0.8B. It is not the exact T4 recipe (Qwen2.5-3B-bnb-4bit, CUDA, 1k Vietnamese Alpaca SFT, and 2k UltraFeedback DPO). The exact UltraFeedback data-prep artifact is present and independently validated; the included adapter metrics disclose the smaller fallback training sample.
