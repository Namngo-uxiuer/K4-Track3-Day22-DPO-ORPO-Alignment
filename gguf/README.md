# Local GGUF deployment artifact

This directory contains the real local export generated from the CPU-fallback
merged model. The binary files are intentionally kept local because GitHub's
normal file limit is 100 MB; the reproducible conversion script, metadata, and
smoke screenshot are committed in the repository. A Git LFS upload was also
attempted, but GitHub rejected new LFS objects for this public fork; therefore
the binaries remain available locally and are not represented by broken public
download pointers.

| File | Size | Purpose |
|---|---:|---|
| `lab22-dpo-fallback-F16.gguf` | ~1.5 GB | Full-precision GGUF intermediate |
| `lab22-dpo-fallback-Q4_K_M.gguf` | 529.3 MB | Quantized deployment artifact |
| `lab22-dpo-fallback-Q5_K_M.gguf` | 578.0 MB | Higher-quality quantized deployment artifact |

The Q4_K_M smoke test passed with the official llama.cpp CPU CLI. Its exact
runtime, SHA-256, prompt, and generated output are recorded in
`data/eval/gguf_smoke.json`; the visual evidence is
`submission/screenshots/06-gguf-smoke.png`.

Scope: this is a real Qwen3.5-0.8B CPU fallback export, not the course's
Qwen2.5-3B CUDA/T4 run. To regenerate it, use
`scripts/merge_cpu_fallback_gguf.py` and `scripts/run_gguf_smoke.py`. The
Q4_K_M and Q5_K_M hashes are also recorded in `data/eval/gguf_manifest.json`.
