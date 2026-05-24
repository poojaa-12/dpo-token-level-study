from __future__ import annotations

import json
import os
from typing import Any

from datasets import load_dataset


def download_ultrafeedback(
    cache_dir: str = "data/raw",
    sample_size: int = 5000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Download UltraFeedback and extract chosen/rejected pairs.
    """
    ds = load_dataset("openbmb/UltraFeedback", split="train")
    ds = ds.shuffle(seed=seed).select(range(sample_size))

    pairs: list[dict[str, Any]] = []
    for row in ds:
        completions = row.get("completions", [])
        if len(completions) < 2:
            continue
        completions = sorted(
            completions,
            key=lambda item: item.get("overall_score", 0),
            reverse=True,
        )
        pairs.append(
            {
                "prompt": row.get("instruction", ""),
                "chosen": completions[0].get("response", ""),
                "rejected": completions[-1].get("response", ""),
                "chosen_score": completions[0].get("overall_score", 0),
                "rejected_score": completions[-1].get("overall_score", 0),
            }
        )

    os.makedirs(cache_dir, exist_ok=True)
    output_path = os.path.join(cache_dir, "ultrafeedback_pairs.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Saved {len(pairs)} pairs to {output_path}")
    return pairs


def load_pairs(path: str = "data/raw/ultrafeedback_pairs.jsonl") -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return pairs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs
