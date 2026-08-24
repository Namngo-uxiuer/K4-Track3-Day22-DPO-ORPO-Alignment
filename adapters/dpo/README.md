# DPO adapter — local CPU fallback

This is a separate LoRA adapter trained from the SFT adapter with the custom
CPU DPO loop used for the local evidence run.

- Base model: `Qwen/Qwen3.5-0.8B`
- LoRA rank: `r=16`
- LoRA alpha: `32`
- β: `0.1`
- Runtime: CPU fallback; not the course T4 Qwen2.5-3B run
- Metrics and data provenance: `dpo_metrics.json`

The exact lab preference artifact is prepared at `data/pref/train.parquet`.
The included adapter discloses that its short CPU demonstration used the
separate `data/pref/cpu_fallback_train.parquet` sample.
