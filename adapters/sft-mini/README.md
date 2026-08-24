# SFT-mini adapter — local CPU fallback

This LoRA adapter is the SFT stage of the locally reproducible fallback run.

- Base model: `Qwen/Qwen3.5-0.8B`
- LoRA rank: `r=16`
- LoRA alpha: `32`
- Runtime: CPU fallback; not the course T4 Qwen2.5-3B run
- Training evidence: `cpu_fallback_training.json`

The course-exact recipe remains in `notebooks/01_sft_mini.py` and requires a
CUDA runtime with the specified Qwen2.5-3B model.
