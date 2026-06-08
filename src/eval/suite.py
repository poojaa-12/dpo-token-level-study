from __future__ import annotations

import json
import os
import hashlib
from typing import Any


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_eval_suite(evals_dir: str = "evals") -> dict[str, list[dict[str, Any]]]:
    return {
        "instruction_following": load_jsonl(os.path.join(evals_dir, "instruction_following.jsonl")),
        "adversarial": load_jsonl(os.path.join(evals_dir, "adversarial.jsonl")),
        "consistency": load_jsonl(os.path.join(evals_dir, "consistency.jsonl")),
    }


def compute_suite_hash(evals_dir: str = "evals") -> str:
    hasher = hashlib.sha256()
    for name in ["instruction_following.jsonl", "adversarial.jsonl", "consistency.jsonl"]:
        path = os.path.join(evals_dir, name)
        with open(path, "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()
