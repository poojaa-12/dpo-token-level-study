from __future__ import annotations

import json
import os
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
