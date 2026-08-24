#!/usr/bin/env python3
"""Update only the Hub README after the repository already exists."""
from __future__ import annotations

import os
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    for path in [REPO / ".env.local", REPO / ".env"]:
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    token = os.environ["HF_TOKEN"]
    repo_id = os.environ["HF_REPO"]
    from huggingface_hub import HfApi

    HfApi(token=token).upload_file(
        path_or_fileobj=str(REPO / "submission" / "HF_MODEL_CARD.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add model card metadata",
    )
    print(f"HF_MODEL_CARD_UPDATE=PASS repo=https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
