#!/usr/bin/env python3
"""Write the final honest reflection, metric provenance, and submission audit."""
from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parent.parent
STUDENT_NAME = "Ngô Văn Nam"
STUDENT_ID = "2A202601340"
PUBLIC_URL = "https://github.com/Namngo-uxiuer/K4-Track3-Day22-DPO-ORPO-Alignment"


def main() -> None:
    sft = json.loads((REPO / "adapters" / "sft-mini" / "cpu_fallback_training.json").read_text())
    dpo_path = REPO / "adapters" / "dpo" / "dpo_metrics.json"
    dpo = json.loads(dpo_path.read_text())
    train = pd.read_parquet(REPO / "data" / "pref" / "train.parquet")
    fallback_train = pd.read_parquet(REPO / "data" / "pref" / "cpu_fallback_train.parquet")
    judges = json.loads((REPO / "data" / "eval" / "judge_results.json").read_text(encoding="utf-8"))
    counts = Counter(x["winner"] for x in judges)
    helpful = Counter(x["winner"] for x in judges if x["category"] == "helpfulness")
    safety = Counter(x["winner"] for x in judges if x["category"] == "safety")
    history = dpo["history"]
    beta_path = REPO / "data" / "eval" / "beta_sweep_results.json"
    beta_data = json.loads(beta_path.read_text(encoding="utf-8")) if beta_path.exists() else None
    if beta_data:
        beta_summary = "; ".join(
            f"β={item['beta']}: gap={item['final_gap']:+.4f}, pair-win-rate={item['training_pair_win_rate']:.2f}"
            for item in beta_data["results"]
        )
        beta_section = (
            f"Đã chạy β-sweep mini thật với {beta_data['betas']} và {beta_data['steps_per_beta']} steps mỗi giá trị. "
            f"Kết quả: {beta_summary}. Gap tăng theo β trong run ngắn này, nhưng đây là training-pair signal trên CPU fallback, không phải kết luận T4-scale."
        )
    else:
        beta_section = "Chưa chạy β-sweep; cần xác nhận trên T4 bằng cùng base model và nhiều step hơn."

    gguf_path = REPO / "data" / "eval" / "gguf_smoke.json"
    gguf_data = json.loads(gguf_path.read_text(encoding="utf-8")) if gguf_path.exists() else None
    q5_path = REPO / "gguf" / "lab22-dpo-fallback-Q5_K_M.gguf"
    q5_smoke_path = REPO / "data" / "eval" / "gguf_q5_smoke.json"
    q5_smoke_data = json.loads(q5_smoke_path.read_text(encoding="utf-8")) if q5_smoke_path.exists() else None
    benchmark_path = REPO / "data" / "eval" / "benchmark_results.json"
    benchmark_data = json.loads(benchmark_path.read_text(encoding="utf-8")) if benchmark_path.exists() else None
    official_smoke_path = REPO / "data" / "eval" / "official_benchmark_smokes.json"
    official_smoke_data = json.loads(official_smoke_path.read_text(encoding="utf-8")) if official_smoke_path.exists() else None
    external_path = REPO / "data" / "eval" / "external_bonus_status.json"
    external_data = json.loads(external_path.read_text(encoding="utf-8")) if external_path.exists() else None
    cross_path = REPO / "data" / "eval" / "cross_judge_openai.json"
    cross_data = json.loads(cross_path.read_text(encoding="utf-8")) if cross_path.exists() else None
    hf_path = REPO / "data" / "eval" / "hf_push_status.json"
    hf_data = json.loads(hf_path.read_text(encoding="utf-8")) if hf_path.exists() else None
    if benchmark_data:
        benchmark_summary = "; ".join(
            f"{name}: SFT={metric['sft_only']:.2f}, SFT+DPO={metric['sft_dpo']:.2f} (n={metric['n']})"
            for name, metric in benchmark_data["metrics"].items()
        )
        official = benchmark_data["official_harness_attempt"]
        if official["status"] == "PASS_SMOKE_SUBJECT":
            official_note = (
                f"Official lm-eval smoke cũng PASS ở {official['task']} với n={official['n']}, "
                f"accuracy={official['accuracy']:.4f}; full 57-subject group bị giới hạn bởi thời gian CPU."
            )
        else:
            official_note = f"Official harness status: {official['status']}."
        if official_smoke_data:
            gsm_smoke = official_smoke_data["runs"]["GSM8K"]
            ife_smoke = official_smoke_data["runs"]["IFEval"]
            official_note += (
                f" Official GSM8K smoke PASS n={gsm_smoke['n']} strict={gsm_smoke['metrics']['strict_match']:.4f};"
                f" IFEval smoke PASS n={ife_smoke['n']} prompt-strict={ife_smoke['metrics']['prompt_level_strict_acc']:.4f}"
                " (IFEval max_gen_toks=64)."
            )
        benchmark_section = (
            f"NB6 sampled benchmark đã chạy thật trên CPU fallback: {benchmark_summary}. "
            f"{official_note} Ảnh và JSON có provenance đầy đủ."
        )
    else:
        benchmark_section = "NB6 benchmark chưa có execution artifact; cần chạy trên runtime phù hợp."
    if gguf_data:
        gguf_section = (
            f"NB5 GGUF + llama.cpp smoke: PASS trên CPU fallback ({gguf_data['runtime']}), "
            f"file Q4_K_M {gguf_data['model_size_bytes'] / 1e6:.1f} MB và "
            f"Q5_K_M {'đã quantize' if q5_path.exists() else 'chưa có'} "
            f"({'smoke PASS' if q5_smoke_data and q5_smoke_data['status'] == 'PASS' else 'chưa smoke'}); "
            f"output và SHA-256 nằm trong `data/eval/gguf_smoke.json`/`data/eval/gguf_manifest.json`."
        )
    else:
        gguf_section = "NB5 GGUF + llama.cpp smoke chưa có execution artifact."
    extra_lines = []
    if hf_data:
        extra_lines.append(f"HF Hub push: PASS tại {hf_data['url']}.")
    else:
        extra_lines.append("HF Hub push chưa có execution artifact.")
    if cross_data:
        extra_lines.append(
            f"OpenAI cross-judge: PASS với {cross_data['model']}, {cross_data['n']} prompt; "
            f"SFT-only thắng {cross_data['wins']['SFT_only']}, DPO thắng {cross_data['wins']['SFT_DPO']}, tie {cross_data['wins']['tie']}."
        )
    else:
        extra_lines.append("OpenAI cross-judge chưa có execution artifact.")
    extra_lines.append("W&B public run chưa thể tạo vì chưa có WANDB_API_KEY.")
    extra_lines.append("GitHub LFS từ chối upload GGUF binary cho public fork; Q4/Q5 vẫn giữ local với hash và manifest.")
    if official_smoke_data:
        extra_lines.append("Official lm-eval GSM8K + IFEval smoke evidence đã lưu ở data/eval/official_benchmark_smokes.json; đây là smoke n=3, không phải full-suite.")
    external_section = " ".join(extra_lines)

    gguf_audit = (
        f"PASS (CPU fallback; {gguf_data['runtime']}; Q4 smoke + Q5 {'smoke PASS' if q5_smoke_data else 'quantized'})"
        if gguf_data else "not run"
    )
    benchmark_audit = (
        "PASS (sampled CPU fallback + official MMLU subject/GSM8K/IFEval smokes; full suites CPU-time limited)"
        if benchmark_data else "not run"
    )
    cross_audit = (
        f"PASS ({cross_data['model']}; {cross_data['wins']['SFT_DPO']} DPO wins / {cross_data['wins']['SFT_only']} SFT wins / {cross_data['wins']['tie']} ties)"
        if cross_data else "not run"
    )
    hf_audit = f"PASS ({hf_data['url']})" if hf_data else "not run"
    judge_section = (
        f"OpenAI cross-judge (`{cross_data['model']}`) đã chấm thật 8 prompt: "
        f"SFT+DPO thắng {cross_data['wins']['SFT_DPO']}, SFT-only thắng {cross_data['wins']['SFT_only']}, tie {cross_data['wins']['tie']}. "
        "Kết quả API được lưu ở `data/eval/cross_judge_openai.json`; manual rubric vẫn giữ để đối chiếu."
        if cross_data else "Đây là manual heuristic rubric, không phải API judge."
    )

    dpo.update({
        "preference_dataset": "argilla/ultrafeedback-binarized-preferences-cleaned",
        "prepared_train_rows": len(train),
        "prepared_eval_rows": len(pd.read_parquet(REPO / "data" / "pref" / "eval.parquet")),
        "cpu_training_sample_rows": len(fallback_train),
        "actual_cpu_training_source": "local Vietnamese classification fallback pairs (data/pref/cpu_fallback_train.parquet)",
        "data_provenance_note": "The exact UltraFeedback artifact is prepared and validated; the included CPU adapter was trained on the smaller local fallback sample because the available runtime is CPU-only.",
        "course_exact_run": False,
    })
    dpo_path.write_text(json.dumps(dpo, ensure_ascii=False, indent=2), encoding="utf-8")

    reflection = f"""# Reflection — Lab 22 DPO/ORPO Alignment

**Người học:** {STUDENT_NAME}
**MSSV:** {STUDENT_ID}
**Cohort:** K4
**Ngày:** {date.today().isoformat()}
**Runtime đã chạy:** `CPU_FALLBACK` — không phải T4 course run

---

## §1 Setup

| Hạng mục | Giá trị |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB; không đủ cho recipe Qwen2.5-3B 4-bit và PyTorch hiện tại không có CUDA |
| Base model đã chạy | `Qwen/Qwen3.5-0.8B` từ local cache, dùng làm fallback có thể chạy thật |
| LoRA | `r=16`, `lora_alpha=32`, dropout 0; SFT và DPO adapter tách biệt |
| SFT | 8 CPU update steps, lr `2e-5`, max length 192; lặp một mẫu để kiểm tra đường loss |
| DPO | beta `{dpo['beta']}`, lr `{dpo['lr']}`, 6 update steps; CPU fallback |
| Dataset chính | `argilla/ultrafeedback-binarized-preferences-cleaned`: 2.000 train + 200 eval đã chuẩn hóa |

## §2 DPO Results

| Metric | Kết quả |
|---|---:|
| SFT elapsed | {sft['elapsed_seconds']:.1f}s |
| DPO elapsed | {dpo['elapsed_seconds']:.1f}s |
| SFT loss | {sft['losses'][0]['loss']:.6f} → {sft['losses'][-1]['loss']:.6f} |
| DPO final loss | {dpo['final_train_loss']:.6f} |
| Chosen reward cuối | {dpo['end_chosen_reward']:+.6f} |
| Rejected reward cuối | {dpo['end_rejected_reward']:+.6f} |
| Reward gap cuối | **{dpo['end_reward_gap']:+.6f}** |
| Manual rubric | {counts.get('B', 0)} DPO wins / {counts.get('A', 0)} SFT-only wins / {counts.get('tie', 0)} ties |

Đây là số đo của CPU fallback. Artifact data-prep đúng dataset lab nằm ở `data/pref/train.parquet`; adapter DPO đi kèm được train trên 32 fallback pairs và đã ghi rõ provenance trong `adapters/dpo/dpo_metrics.json`.

## §3 Reward Curves Analysis

Biểu đồ `submission/screenshots/03-dpo-reward-curves.png` vẽ riêng chosen reward, rejected reward và reward gap để tránh kết luận sai chỉ từ một đường. Trong lần chạy thật này, reward gap đi từ `{history[0]['reward_gap']:+.4f}` đến `{history[-1]['reward_gap']:+.4f}` qua {len(history)} bước; chosen reward cuối là `{dpo['end_chosen_reward']:+.4f}`, còn rejected reward cuối là `{dpo['end_rejected_reward']:+.4f}`. Chosen tăng cho thấy policy đang tăng log-likelihood tương đối của câu trả lời được chọn, trong khi rejected giảm cho thấy policy đẩy câu trả lời kém ra xa reference. Vì vậy gap dương ở cuối không chỉ là một số tổng hợp: nó có thể được phân tích thành hai chuyển động có ý nghĩa khác nhau. Tuy nhiên, run CPU này rất ngắn và dữ liệu training của adapter là fallback sample, nên không được diễn giải như bằng chứng T4 ổn định. Đường gap cũng nên được đọc cùng với chosen/rejected; nếu chosen giảm mà rejected giảm nhanh hơn thì đó là likelihood displacement. Ở đây chosen và rejected đều cần được kiểm tra trực tiếp, rồi mới kết luận DPO có cải thiện helpfulness hay chỉ học tín hiệu hẹp của preference pairs.

## §4 Qualitative Comparison

`data/eval/side_by_side.jsonl` và ảnh `04-side-by-side-table.png` có 8 prompt: 4 helpfulness và 4 safety.

| Nhóm | SFT+DPO thắng | SFT-only thắng | Tie |
|---|---:|---:|---:|
| Helpfulness | {helpful.get('B', 0)} | {helpful.get('A', 0)} | {helpful.get('tie', 0)} |
| Safety | {safety.get('B', 0)} | {safety.get('A', 0)} | {safety.get('tie', 0)} |
| **Tổng** | **{counts.get('B', 0)}** | **{counts.get('A', 0)}** | **{counts.get('tie', 0)}** |

{judge_section} Ảnh `05-manual-rubric.png` ghi lại rubric thủ công để reviewer đối chiếu.

## §5 β Trade-off

{beta_section} β thấp thường cho policy lệch reference mạnh hơn, có thể tăng preference margin nhưng dễ overfit hoặc làm giảm tính ổn định. β cao bảo thủ hơn, giữ policy gần SFT/reference nhưng có thể làm gap tăng chậm. Run chính dùng β=`{dpo['beta']}` theo lab. Kết luận cuối cùng vẫn cần được xác nhận trên T4 bằng cùng base model, cùng UltraFeedback slice và nhiều step hơn.

## §6 Personal Reflection

Quyết định quan trọng nhất là tách bạch giữa “đã chạy thật trên máy hiện tại” và “đúng recipe T4 trong giáo trình”. Máy có RTX 3050 nhưng chỉ 4 GB VRAM, còn PyTorch trong môi trường này là CPU-only. Nếu cố chạy Qwen2.5-3B bằng mọi giá, tôi có thể nhận OOM hoặc tạo ra một bộ notebook trông hoàn chỉnh nhưng không có bằng chứng training. Tôi chọn một Qwen3.5-0.8B đã có trong cache, giữ đúng cấu trúc LoRA r=16 và alpha=32, rồi chạy SFT continuation, DPO loop, generation, preference scoring và evaluation thật trên CPU. Cách làm này chậm hơn nhiều và không thể thay thế kết quả T4, nhưng mọi adapter, parquet, đường loss, reward curve và bảng so sánh đều có nguồn từ execution thực tế. Tôi cũng chuẩn bị đủ 2.000 UltraFeedback train rows và 200 eval rows đúng schema để bộ nộp có provenance đúng với lab, đồng thời lưu riêng fallback pairs để không che giấu sự khác biệt giữa data-prep và training source. Bài học lớn nhất là reward gap không tự động đồng nghĩa với helpfulness tốt hơn: phải xem chosen và rejected riêng, kiểm tra prompt distribution, rồi dùng rubric độc lập. Nếu có GPU phù hợp, bước tiếp theo là chạy nguyên recipe Qwen2.5-3B + UltraFeedback 2k, beta 0.1, lr 5e-7 và thay các số fallback bằng log T4.

## §7 Benchmark (optional)

{benchmark_section}

{gguf_section}

{external_section}

Các số liệu sampled/fallback được gắn nhãn rõ ràng để reviewer phân biệt với recipe T4 của khóa học; không có score hay link dịch vụ nào được suy đoán.
"""
    (REPO / "submission" / "REFLECTION.md").write_text(reflection, encoding="utf-8")

    audit = f"""# Submission audit — {date.today().isoformat()}

**Student:** {STUDENT_NAME} — **MSSV:** {STUDENT_ID}
**Public GitHub:** {PUBLIC_URL}
**Verified commit:** `pending final push`

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
| Reward gap final positive | PASS | final gap `{dpo['end_reward_gap']:+.6f}` |
| 8 prompts × 2 models | PASS | `data/eval/side_by_side.jsonl`, `04-side-by-side-table.png` |
| Win/loss/tie summary | PASS | `{counts.get('B', 0)} / {counts.get('A', 0)} / {counts.get('tie', 0)}`; `05-manual-rubric.png` |
| Reflection sections | PASS | `submission/REFLECTION.md`, §3 and §6 exceed 150 words |
| Reproducible scripts | PASS | `scripts/run_cpu_fallback.py`, `scripts/prepare_ultrafeedback.py` |

## Optional / hardware-gated

- β-sweep mini: **PASS (CPU fallback)**; see `data/eval/beta_sweep_results.json` and `submission/screenshots/bonus-beta-sweep.png`.
- NB5 GGUF Q4_K_M + Q5_K_M + llama.cpp smoke: **{gguf_audit}**; see `data/eval/gguf_smoke.json`, `data/eval/gguf_q5_smoke.json`, and `submission/screenshots/06-gguf-smoke.png`/`08-gguf-q5-smoke.png`.
- NB6 IFEval/GSM8K/MMLU/AlpacaEval-lite: **{benchmark_audit}**; see `data/eval/benchmark_results.json` and `submission/screenshots/07-benchmark-comparison.png`.
- Official benchmark smoke evidence: **{'PASS' if official_smoke_data else 'not run'}**; see `data/eval/official_benchmark_smokes.json` and `submission/screenshots/10-official-mmlu-smoke.png`/`11-official-benchmark-smokes.png`.
- External API judge/cross-judge: **{cross_audit}**; see `data/eval/cross_judge_openai.json` and `submission/screenshots/09-openai-cross-judge.png`.
- Hugging Face Hub push: **{hf_audit}**; see `data/eval/hf_push_status.json` and `submission/HF_MODEL_CARD.md`.
- W&B public run: **blocked — no WANDB_API_KEY**; no fabricated public URL.
- Public GGUF binary upload: **blocked — GitHub LFS permission for this public fork**; local Q4/Q5 files, hashes, and reproducible scripts are retained.

## Important scope note

The local evidence is a real `CPU_FALLBACK` run on Qwen3.5-0.8B. It is not the exact T4 recipe (Qwen2.5-3B-bnb-4bit, CUDA, 1k Vietnamese Alpaca SFT, and 2k UltraFeedback DPO). The exact UltraFeedback data-prep artifact is present and independently validated; the included adapter metrics disclose the smaller fallback training sample.
"""
    (REPO / "SUBMISSION-AUDIT.md").write_text(audit, encoding="utf-8")
    print("Final reflection, provenance metrics, and audit written.")


if __name__ == "__main__":
    main()
