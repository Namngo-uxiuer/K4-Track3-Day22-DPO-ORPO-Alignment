# CPU fallback run

The course path is `Qwen2.5-3B-bnb-4bit` + Unsloth + CUDA/T4. This machine has
an RTX 3050 Laptop GPU with 4 GB VRAM and a CPU-only PyTorch wheel, so the
course path cannot run locally.

The repository now contains a real CPU fallback run using the cached
`Qwen/Qwen3.5-0.8B` model, a compatible Day21 LoRA adapter, 32 local Vietnamese
classification fallback pairs, and a small custom DPO loop. Separately, the
exact lab preference artifact has also been prepared from
`argilla/ultrafeedback-binarized-preferences-cleaned` with 2,000 train and 200
eval rows.

The run produced:

- SFT LoRA: `adapters/sft-mini/` (`r=16`, `lora_alpha=32`)
- DPO LoRA: `adapters/dpo/` (`r=16`, `lora_alpha=32`)
- Exact preference data: `data/pref/train.parquet` (2,000 pairs) and `eval.parquet` (200 pairs)
- Fallback training data: `data/pref/cpu_fallback_train.parquet` (32 pairs)
- Evaluation: `data/eval/side_by_side.jsonl` and `judge_results.json`
- Screenshots: six evidence PNGs under `submission/screenshots/`
- Reflection: `submission/REFLECTION.md`

Run it again on this machine with:

```powershell
$env:TRANSFORMERS_OFFLINE = "1"
& ".venv-cpu\Scripts\python.exe" -u scripts\run_cpu_fallback.py
```

The included adapter is trained on the fallback pairs and its metrics disclose
that fact. The exact data-prep artifact is present, but the fallback is not
equivalent to the course T4/Qwen2.5-3B/UltraFeedback training experiment. Use
`colab/Lab22_DPO_T4.ipynb` on a T4 to replace it with the course-exact run.
