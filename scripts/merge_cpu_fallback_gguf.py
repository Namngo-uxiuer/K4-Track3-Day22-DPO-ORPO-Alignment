#!/usr/bin/env python3
"""Merge the local SFT+DPO adapters and export a real GGUF fallback."""
from __future__ import annotations

import gc
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get(
    "CPU_MODEL_PATH",
    r"C:\Users\Acer\.cache\huggingface\hub\models--Qwen--Qwen3.5-0.8B\snapshots\2fc06364715b967f1860aea9cf38778875588b17",
))
MERGED = Path(os.environ.get("MERGED_OUTPUT", str(Path(os.environ.get("TEMP", str(REPO))) / "lab22-merged-cpu-fallback")))
GGUF_DIR = REPO / "gguf"
CONVERTER = Path(os.environ.get("LLAMA_CONVERTER", str(Path(os.environ.get("TEMP", "")) / "llama.cpp-lab22" / "convert_hf_to_gguf.py")))
QUANTIZE = Path(os.environ.get("LLAMA_QUANTIZE", str(Path(os.environ.get("TEMP", "")) / "llama-b10605-cpu" / "llama-quantize.exe")))


def main() -> None:
    if MERGED.exists():
        shutil.rmtree(MERGED)
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading base: {MODEL_PATH}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cpu", local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    sft = PeftModel.from_pretrained(base, REPO / "adapters" / "sft-mini")
    merged_sft = sft.merge_and_unload()
    del sft, base
    gc.collect()
    dpo = PeftModel.from_pretrained(merged_sft, REPO / "adapters" / "dpo")
    merged = dpo.merge_and_unload()
    del dpo, merged_sft
    gc.collect()
    if hasattr(merged.config, "tie_word_embeddings"):
        merged.config.tie_word_embeddings = False
    merged.save_pretrained(MERGED, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(MERGED)
    del merged, tokenizer
    gc.collect()
    print(f"Merged model: {MERGED}", flush=True)

    if not CONVERTER.exists():
        raise FileNotFoundError(f"Missing converter: {CONVERTER}")
    if not QUANTIZE.exists():
        raise FileNotFoundError(f"Missing quantizer: {QUANTIZE}")
    f16 = GGUF_DIR / "lab22-dpo-fallback-F16.gguf"
    q4 = GGUF_DIR / "lab22-dpo-fallback-Q4_K_M.gguf"
    q5 = GGUF_DIR / "lab22-dpo-fallback-Q5_K_M.gguf"
    # The cached Qwen3.5 checkpoint carries an MTP config flag but the merged
    # fallback has no MTP tensors.  Excluding the optional speculative head
    # keeps block_count aligned with the 24 exported transformer blocks.
    cmd = [sys.executable, str(CONVERTER), str(MERGED), "--outfile", str(f16), "--outtype", "f16", "--no-mtp"]
    print("Converting HF -> GGUF F16", flush=True)
    subprocess.run(cmd, check=True)
    print("Quantizing GGUF -> Q4_K_M", flush=True)
    subprocess.run([str(QUANTIZE), str(f16), str(q4), "Q4_K_M"], check=True)
    print("Quantizing GGUF -> Q5_K_M", flush=True)
    subprocess.run([str(QUANTIZE), str(f16), str(q5), "Q5_K_M"], check=True)
    print(f"GGUF ready: {q4} ({q4.stat().st_size / 1e6:.1f} MB)", flush=True)
    print(f"GGUF ready: {q5} ({q5.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
