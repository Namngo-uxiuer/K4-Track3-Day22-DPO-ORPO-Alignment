#!/usr/bin/env python3
"""Run a real, explicitly sampled CPU benchmark for the local fallback models.

This is not a replacement for the full official harness.  It exists so NB6 has
auditable execution evidence when the official datasets cannot be downloaded in
the restricted runtime.
"""
from __future__ import annotations

import gc
import json
import os
import re
import time
from pathlib import Path

MPL_CACHE = Path(__file__).resolve().parent.parent / ".matplotlib-cache"
MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from peft import PeftModel

from run_cpu_fallback import EVAL_OUT, MODEL_PATH, SFT_OUT, DPO_OUT, generate, load_base


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "eval" / "benchmark_results.json"
SHOT = REPO / "submission" / "screenshots" / "07-benchmark-comparison.png"

CASES = {
    "IFEval-lite": [
        {
            "id": "ifeval-1",
            "prompt": "Trả lời bằng đúng 3 gạch đầu dòng, mỗi gạch đầu dòng nói một lợi ích của kiểm thử mô hình.",
            "check": "three_bullets",
        },
        {
            "id": "ifeval-2",
            "prompt": "Viết đúng 2 câu tiếng Việt giải thích vì sao cần kiểm tra dữ liệu trước khi huấn luyện.",
            "check": "two_sentences",
        },
        {
            "id": "ifeval-3",
            "prompt": "Chỉ trả về JSON hợp lệ có hai khóa answer và reason; không thêm markdown.",
            "check": "json_keys",
        },
    ],
    "GSM8K-lite": [
        {"id": "gsm8k-1", "prompt": "Một quyển vở giá 3 đô la. Mua 4 quyển hết bao nhiêu đô la? Trả lời số cuối cùng.", "answer": "12"},
        {"id": "gsm8k-2", "prompt": "Lan có 12 quả táo và cho đi 5 quả. Lan còn lại bao nhiêu quả? Trả lời số cuối cùng.", "answer": "7"},
        {"id": "gsm8k-3", "prompt": "Một xe chạy 60 km mỗi giờ trong 2,5 giờ. Xe đi được bao nhiêu km? Trả lời số cuối cùng.", "answer": "150"},
    ],
    "MMLU-lite": [
        {"id": "mmlu-1", "prompt": "Chọn đúng một đáp án. 2 + 2 bằng bao nhiêu? A. 3 B. 4 C. 5 D. 6. Chỉ trả lời A, B, C hoặc D.", "answer": "B"},
        {"id": "mmlu-2", "prompt": "Chọn đúng một đáp án. Ngôn ngữ nào thường dùng để tạo trang web? A. SQL B. Bash C. JavaScript D. YAML. Chỉ trả lời A, B, C hoặc D.", "answer": "C"},
        {"id": "mmlu-3", "prompt": "Chọn đúng một đáp án. Nước sôi ở khoảng bao nhiêu độ C ở mực nước biển? A. 0 B. 50 C. 100 D. 200. Chỉ trả lời A, B, C hoặc D.", "answer": "C"},
    ],
}


def score_ifeval(case: dict, output: str) -> int:
    text = output.strip()
    if case["check"] == "three_bullets":
        return int(len(re.findall(r"(?:^|\n)\s*(?:[-*•]|\d+[.)])\s+", text)) == 3)
    if case["check"] == "two_sentences":
        return int(len(re.findall(r"[^.!?]+[.!?]", text)) == 2)
    try:
        value = json.loads(text)
        return int(set(value) >= {"answer", "reason"})
    except (json.JSONDecodeError, TypeError):
        return 0


def score_gsm8k(case: dict, output: str) -> int:
    return int(case["answer"] in output.strip().split()[-12:])


def score_mmlu(case: dict, output: str) -> int:
    choices = re.findall(r"\b([ABCD])\b", output.upper())
    return int(bool(choices) and choices[-1] == case["answer"])


def generate_suite(model, tokenizer) -> dict:
    all_cases = [case for cases in CASES.values() for case in cases]
    outputs = {}
    for case in all_cases:
        print(f"  generating {case['id']}", flush=True)
        outputs[case["id"]] = generate(model, tokenizer, case["prompt"], max_new_tokens=64)
    return outputs


def load_models() -> tuple[dict, dict]:
    base, tokenizer = load_base()
    sft = PeftModel.from_pretrained(base, SFT_OUT, is_trainable=False)
    sft_outputs = generate_suite(sft, tokenizer)
    del sft, base
    gc.collect()
    base, tokenizer = load_base()
    sft = PeftModel.from_pretrained(base, SFT_OUT, is_trainable=False)
    merged = sft.merge_and_unload()
    dpo = PeftModel.from_pretrained(merged, DPO_OUT, is_trainable=False)
    dpo_outputs = generate_suite(dpo, tokenizer)
    del dpo, merged, sft, base
    gc.collect()
    return sft_outputs, dpo_outputs


def score_suite(outputs: dict, suite: str, cases: list[dict]) -> float:
    scores = []
    for case in cases:
        if suite == "IFEval-lite":
            scores.append(score_ifeval(case, outputs[case["id"]]))
        elif suite == "GSM8K-lite":
            scores.append(score_gsm8k(case, outputs[case["id"]]))
        else:
            scores.append(score_mmlu(case, outputs[case["id"]]))
    return sum(scores) / len(scores)


def main() -> None:
    started = time.perf_counter()
    print(f"CPU fallback benchmark using {MODEL_PATH}", flush=True)
    sft_outputs, dpo_outputs = load_models()
    metrics = {}
    for suite, cases in CASES.items():
        metrics[suite] = {
            "sft_only": score_suite(sft_outputs, suite, cases),
            "sft_dpo": score_suite(dpo_outputs, suite, cases),
            "n": len(cases),
            "method": "programmatic_grader_on_local_cpu_fallback_outputs",
        }

    judges = json.loads((EVAL_OUT / "judge_results.json").read_text(encoding="utf-8"))
    wins_sft = sum(item["winner"] == "A" for item in judges)
    wins_dpo = sum(item["winner"] == "B" for item in judges)
    ties = sum(item["winner"] == "tie" for item in judges)
    metrics["AlpacaEval-lite"] = {
        "sft_only": (wins_sft + 0.5 * ties) / len(judges),
        "sft_dpo": (wins_dpo + 0.5 * ties) / len(judges),
        "n": len(judges),
        "method": "manual_pairwise_rubric_from_NB4",
    }
    payload = {
        "status": "PASS_SAMPLED_FALLBACK",
        "official_full_suite": False,
        "compute_tier": "CPU_FALLBACK",
        "base_model": "Qwen/Qwen3.5-0.8B",
        "model_source": str(MODEL_PATH),
        "official_harness_attempt": {
            "status": "BLOCKED_DATASET_DOWNLOAD",
            "command": "lm-eval run --model hf --tasks mmlu --limit 1 --device cpu",
            "reason": "Hugging Face dataset cais/mmlu blocked by restricted socket/network runtime after the model loaded successfully.",
        },
        "metrics": metrics,
        "outputs": {"sft_only": sft_outputs, "sft_dpo": dpo_outputs},
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "note": "These are real sampled fallback scores, not official full IFEval/GSM8K/MMLU scores and not T4-scale evidence.",
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    names = list(metrics)
    sft_scores = [metrics[name]["sft_only"] for name in names]
    dpo_scores = [metrics[name]["sft_dpo"] for name in names]
    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=160)
    x = list(range(len(names)))
    width = 0.34
    b1 = ax.bar([i - width / 2 for i in x], sft_scores, width, label="SFT-only", color="#6b8bb5")
    b2 = ax.bar([i + width / 2 for i in x], dpo_scores, width, label="SFT+DPO", color="#1f9d78")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for i, name in enumerate(names):
        delta = dpo_scores[i] - sft_scores[i]
        ax.text(i, max(sft_scores[i], dpo_scores[i]) + 0.13, f"Δ {delta:+.2f}", ha="center", fontsize=10, color="#334155")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score / win rate")
    ax.set_title("")
    fig.suptitle("NB6 — Benchmark comparison (real sampled CPU fallback)",
                 x=0.095, y=0.98, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.095, 0.925,
             "Official full datasets were network-blocked; values below are auditable local samples.",
             fontsize=10, color="#9a3412")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.07, right=0.98)
    fig.savefig(SHOT, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"metrics": metrics, "elapsed_seconds": payload["elapsed_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
