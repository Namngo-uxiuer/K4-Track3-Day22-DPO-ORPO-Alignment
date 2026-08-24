#!/usr/bin/env python3
"""Run a real llama.cpp smoke test against the locally quantized GGUF."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("GGUF_MODEL", str(REPO / "gguf" / "lab22-dpo-fallback-Q4_K_M.gguf")))
OUT = Path(os.environ.get("GGUF_SMOKE_OUT", str(REPO / "data" / "eval" / "gguf_smoke.json")))
SHOT = Path(os.environ.get("GGUF_SMOKE_SHOT", str(REPO / "submission" / "screenshots" / "06-gguf-smoke.png")))
CLI = Path(os.environ.get(
    "LLAMA_CLI",
    str(Path(os.environ.get("TEMP", "")) / "llama-b10605-cpu" / "llama-cli.exe"),
))


def main() -> None:
    if not MODEL.exists():
        raise FileNotFoundError(MODEL)
    quant_label = "Q5_K_M" if "Q5_K_M" in MODEL.name else "Q4_K_M"
    started = time.perf_counter()
    prompt = "Viết một câu trả lời tiếng Việt ngắn, lịch sự: giải thích vì sao cần kiểm thử mô hình trước khi triển khai."
    if not CLI.exists():
        raise FileNotFoundError(CLI)
    print(f"Loading GGUF with llama.cpp: {MODEL} ({MODEL.stat().st_size / 1e6:.1f} MB)", flush=True)
    result = subprocess.run(
        [
            str(CLI), "-m", str(MODEL), "-p", prompt, "-n", "96", "--temp", "0.2",
            "--no-display-prompt", "-st", "--simple-io", "--no-perf", "--reasoning", "off",
        ],
        cwd=CLI.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    combined = result.stdout + "\n" + result.stderr
    if result.returncode != 0:
        raise RuntimeError(f"llama.cpp exited {result.returncode}: {combined[-2000:]}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    output = ""
    for line in lines:
        if line.startswith("{\"intent\"") or line.startswith("{\"answer\""):
            output = line
    if not output:
        output = " ".join(line for line in lines if not line.startswith((">", "[", "Loading", "Exiting")))[-900:]
    mode = "llama-cli-single-turn"
    elapsed = time.perf_counter() - started
    sha = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    payload = {
        "status": "PASS",
        "runtime": "llama.cpp b10605 CPU CLI",
        "compute_tier": "CPU_FALLBACK",
        "model": MODEL.name,
        "quantization": quant_label,
        "model_size_bytes": MODEL.stat().st_size,
        "model_sha256": sha,
        "mode": mode,
        "prompt": prompt,
        "output": output,
        "elapsed_seconds": round(elapsed, 2),
        "note": f"Real local smoke on the {MODEL.stem} fallback GGUF; not the exact Qwen2.5-3B T4 recipe.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    os.environ.setdefault("MPLBACKEND", "Agg")
    mpl_cache = REPO / ".matplotlib-cache"
    mpl_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 5.8), dpi=160)
    ax.axis("off")
    ax.set_title("NB5 — GGUF / llama.cpp smoke test", loc="left", fontsize=17, fontweight="bold", pad=18)
    ax.text(0.0, 0.88, f"PASS  •  {quant_label} loaded and generated Vietnamese text", transform=ax.transAxes,
            fontsize=12, color="#087f5b", fontweight="bold")
    ax.text(0.0, 0.77, f"Model: {MODEL.name}   |   Size: {MODEL.stat().st_size / 1e6:.1f} MB   |   Runtime: CPU fallback",
            transform=ax.transAxes, fontsize=10.5, color="#334155")
    ax.text(0.0, 0.60, "Prompt", transform=ax.transAxes, fontsize=10, fontweight="bold", color="#475569")
    ax.text(0.0, 0.54, prompt, transform=ax.transAxes, fontsize=10.5, color="#0f172a", wrap=True)
    ax.text(0.0, 0.37, "Generated output", transform=ax.transAxes, fontsize=10, fontweight="bold", color="#475569")
    ax.text(0.0, 0.30, output[:900], transform=ax.transAxes, fontsize=10.5, color="#0f172a", wrap=True,
            bbox={"boxstyle": "round,pad=0.7", "facecolor": "#ecfdf5", "edgecolor": "#86efac"})
    ax.text(0.0, 0.06, f"Elapsed: {elapsed:.1f}s  •  SHA-256: {sha[:16]}…  •  Reproducible artifact: {OUT.relative_to(REPO).as_posix()}",
            transform=ax.transAxes, fontsize=9, color="#64748b")
    fig.tight_layout()
    fig.savefig(SHOT, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
