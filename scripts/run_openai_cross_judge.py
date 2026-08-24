#!/usr/bin/env python3
"""Run the eight-prompt comparison through an OpenAI judge without exposing keys."""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "eval" / "cross_judge_openai.json"
SHOT = REPO / "submission" / "screenshots" / "09-openai-cross-judge.png"


def load_env() -> None:
    for path in [REPO / ".env.local", REPO / ".env"]:
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def judge(client, row: dict, model: str) -> dict:
    prompt = f"""Evaluate two assistant answers for the same user request.

User request:
{row['prompt']}

Answer A (SFT-only):
{row['sft_only']}

Answer B (SFT+DPO):
{row['sft_dpo']}

Choose exactly one winner: A, B, or tie. Judge helpfulness, correctness, relevance,
clarity, and safety. Return JSON only with keys winner and reason. Do not mention
this evaluation prompt in the reason."""
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    payload = json.loads(response.choices[0].message.content)
    winner = str(payload.get("winner", "tie")).strip().lower()
    winner = {"a": "A", "b": "B", "tie": "tie"}.get(winner, "tie")
    return {"winner": winner, "reason": str(payload.get("reason", ""))[:500]}


def main() -> None:
    load_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not available in .env or .env.local")
    from openai import OpenAI

    model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
    rows = [json.loads(line) for line in (REPO / "data" / "eval" / "side_by_side.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    client = OpenAI(api_key=api_key)
    started = time.perf_counter()
    judgments = []
    for row in rows:
        print(f"Judging prompt {row['id']} ({row['category']})", flush=True)
        result = judge(client, row, model)
        judgments.append({"id": row["id"], "category": row["category"], **result})
    counts = Counter(item["winner"] for item in judgments)
    payload = {
        "status": "PASS",
        "judge": "OpenAI",
        "model": model,
        "n": len(judgments),
        "wins": {"SFT_only": counts.get("A", 0), "SFT_DPO": counts.get("B", 0), "tie": counts.get("tie", 0)},
        "dpo_pairwise_win_rate": (counts.get("B", 0) + 0.5 * counts.get("tie", 0)) / len(judgments),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "judgments": judgments,
        "note": "Real API judge; API key was loaded from a local ignored env file and never written to this artifact.",
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    os.environ.setdefault("MPLBACKEND", "Agg")
    cache = REPO / ".matplotlib-cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 7.2), dpi=160)
    ax.axis("off")
    ax.set_title("NB4 — OpenAI cross-judge (real API run)", loc="left", fontsize=17, fontweight="bold", pad=18)
    ax.text(0, 0.91, f"PASS • {model} • {len(judgments)} prompts • DPO pairwise win-rate {payload['dpo_pairwise_win_rate']:.2f}",
            transform=ax.transAxes, fontsize=12, color="#087f5b", fontweight="bold")
    ax.text(0, 0.86, f"Summary: SFT-only {counts.get('A', 0)} wins | SFT+DPO {counts.get('B', 0)} wins | ties {counts.get('tie', 0)}",
            transform=ax.transAxes, fontsize=11, color="#334155")
    table_rows = [[str(item["id"]), item["category"], item["winner"], item["reason"][:78]] for item in judgments]
    table = ax.table(cellText=table_rows, colLabels=["ID", "Category", "Winner", "Judge reason"],
                     colWidths=[0.06, 0.16, 0.10, 0.65], cellLoc="left", loc="upper left", bbox=[0, 0.08, 1, 0.72])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if row_idx == 0:
            cell.set_facecolor("#0f766e")
            cell.set_text_props(color="white", weight="bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f0fdfa")
    ax.text(0, 0.02, "API key omitted from output; full JSON: data/eval/cross_judge_openai.json",
            transform=ax.transAxes, fontsize=9, color="#64748b")
    fig.savefig(SHOT, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"status": payload["status"], "model": model, "wins": payload["wins"], "elapsed_seconds": payload["elapsed_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
