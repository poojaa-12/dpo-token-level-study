from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any


def write_results_json(results: dict[str, Any], output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return path


def write_jsonl(rows: list[dict[str, Any]], output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def write_run_manifest(manifest: dict[str, Any], output_dir: str, filename: str = "run_manifest.json") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    payload = {
        **manifest,
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def write_summary(results_by_model: dict[str, dict[str, float]], output_dir: str = "results") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "summary.md")
    lines = [
        "# Evaluation Summary",
        "",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "",
        "| Model | Instruction Following | Refusal Rate | Consistency |",
        "|---|---:|---:|---:|",
    ]
    for model_name, metrics in results_by_model.items():
        lines.append(
            f"| {model_name} | {metrics['instruction_following']:.2%} | {metrics['refusal_rate']:.2%} | {metrics['consistency']:.2%} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path
