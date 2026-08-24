#!/usr/bin/env python3
"""Push the local DPO adapter and model card to Hugging Face Hub."""
from __future__ import annotations

import os
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


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


def main() -> None:
    load_env()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    repo_id = os.environ.get("HF_REPO")
    if not token or not repo_id or "<" in repo_id:
        raise RuntimeError("HF_TOKEN and HF_REPO must be set in .env or .env.local")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(REPO / "adapters" / "dpo"),
        path_in_repo="adapter",
        commit_message="Upload Lab 22 DPO adapter and metrics",
    )
    api.upload_file(
        path_or_fileobj=str(REPO / "submission" / "HF_MODEL_CARD.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add Lab 22 model card",
    )
    print(f"HF_PUSH_STATUS=PASS repo=https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
