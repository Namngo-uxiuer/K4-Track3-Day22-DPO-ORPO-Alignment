#!/usr/bin/env python3
"""Run a small, real CPU β sweep without overwriting the submitted DPO adapter."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

import run_cpu_fallback as fallback
from peft import LoraConfig, PeftModel, get_peft_model


REPO = Path(__file__).resolve().parent.parent
BETAS = [0.05, 0.1, 0.5]
STEPS = 3


def run_one(beta: float, rows: list[dict], tokenizer) -> dict:
    base, _ = fallback.load_base()
    sft_model = PeftModel.from_pretrained(base, fallback.SFT_OUT, is_trainable=False)
    merged = sft_model.merge_and_unload()
    target_modules = json.loads((fallback.SFT_OUT / "adapter_config.json").read_text()).get("target_modules")
    model = get_peft_model(merged, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=target_modules,
    ))
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=fallback.DPO_LR)
    history = []
    for step in range(STEPS):
        row = rows[step % len(rows)]
        batch, starts = fallback.seq_batch(tokenizer, [(row["prompt"], row["chosen"]), (row["prompt"], row["rejected"])])
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            with model.disable_adapter():
                ref_lp = fallback.sequence_logprobs(model, batch, starts)
        policy_lp = fallback.sequence_logprobs(model, batch, starts)
        rewards = beta * (policy_lp.detach() - ref_lp)
        margin = beta * ((policy_lp[0] - policy_lp[1]) - (ref_lp[0] - ref_lp[1]))
        loss = -F.logsigmoid(margin)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        history.append({
            "step": step + 1,
            "loss": float(loss.detach()),
            "chosen_reward": float(rewards[0]),
            "rejected_reward": float(rewards[1]),
            "reward_gap": float(rewards[0] - rewards[1]),
        })
        print(f"beta={beta} step={step + 1}/{STEPS} gap={history[-1]['reward_gap']:+.5f}", flush=True)
    gaps = [x["reward_gap"] for x in history]
    usable = gaps[1:] if len(gaps) > 1 else gaps
    result = {
        "beta": beta,
        "steps": STEPS,
        "history": history,
        "initial_gap": gaps[0],
        "final_gap": gaps[-1],
        "training_pair_win_rate": sum(x > 0 for x in usable) / len(usable),
        "runtime": "CPU_FALLBACK",
        "note": "Mini sweep on the same local fallback preference sample; not a T4-scale conclusion.",
    }
    del model, optimizer, merged, sft_model, base
    gc.collect()
    return result


def main() -> None:
    torch.set_num_threads(4)
    rows = pd.read_parquet(REPO / "data" / "pref" / "cpu_fallback_train.parquet").to_dict("records")
    tokenizer = fallback.load_base(tokenizer_only=True)
    results = [run_one(beta, rows, tokenizer) for beta in BETAS]
    out = REPO / "data" / "eval" / "beta_sweep_results.json"
    out.write_text(json.dumps({
        "betas": BETAS,
        "steps_per_beta": STEPS,
        "results": results,
        "student": "Ngô Văn Nam",
        "mssv": "2A202601340",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    gaps = [x["final_gap"] for x in results]
    wins = [x["training_pair_win_rate"] for x in results]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(BETAS, gaps, marker="o", color="#2e548a")
    axes[0].set_xlabel("β")
    axes[0].set_ylabel("Final reward gap")
    axes[0].set_title("Final reward gap vs β")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(BETAS, wins, marker="o", color="#c83538")
    axes[1].set_xlabel("β")
    axes[1].set_ylabel("Training-pair win rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Training-pair win rate vs β")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("β sweep mini-experiment · CPU fallback", y=1.02)
    fig.tight_layout()
    fig.savefig(REPO / "submission" / "screenshots" / "bonus-beta-sweep.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"results": results, "plot": "submission/screenshots/bonus-beta-sweep.png"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
