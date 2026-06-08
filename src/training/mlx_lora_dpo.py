from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.data.preprocess import format_instruction_prompt


def _validate_config(config: dict[str, Any]) -> None:
    required = ["model_name", "dataset_path", "adapter_path", "iters"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required MLX config keys: {missing}")

    dataset_path = Path(config["dataset_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. Generate it with run_analysis.py --preprocess."
        )
    if dataset_path.stat().st_size == 0:
        raise ValueError(f"Dataset is empty: {dataset_path}")


def _prepare_mlx_train_data(
    dataset_path: str,
    output_dir: str,
    max_rows: int | None = None,
) -> str:
    """
    Convert preference dataset into MLX LoRA train.jsonl format.
    Uses chosen responses as supervised targets to stay memory efficient.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"

    rows_written = 0
    with open(dataset_path, "r", encoding="utf-8") as src, open(
        train_path, "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if max_rows is not None and rows_written >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            pair = json.loads(line)
            prompt = pair.get("prompt")
            chosen = pair.get("chosen")
            if prompt is None or chosen is None:
                # Already tokenized dataset fallback: skip row.
                continue
            payload = {
                "prompt": format_instruction_prompt(prompt),
                "completion": chosen,
            }
            dst.write(json.dumps(payload, ensure_ascii=False) + "\n")
            rows_written += 1

    if rows_written == 0:
        raise ValueError(
            "No usable prompt/chosen rows were found. Rebuild raw/preprocessed data first."
        )
    return str(train_path)


def train_mlx_lora(config: dict[str, Any]) -> dict[str, Any]:
    """
    Launch MLX LoRA fine-tuning via mlx_lm CLI.
    """
    _validate_config(config)

    adapter_path = Path(config["adapter_path"])
    adapter_path.mkdir(parents=True, exist_ok=True)
    mlx_data_dir = adapter_path / "mlx_data"
    train_jsonl = _prepare_mlx_train_data(
        dataset_path=config["dataset_path"],
        output_dir=str(mlx_data_dir),
        max_rows=config.get("max_rows"),
    )

    command = [
        "python3",
        "-m",
        "mlx_lm.lora",
        "--model",
        str(config["model_name"]),
        "--train",
        "--data",
        str(mlx_data_dir),
        "--adapter-path",
        str(adapter_path),
        "--iters",
        str(int(config.get("iters", 300))),
        "--batch-size",
        str(int(config.get("batch_size", 1))),
        "--learning-rate",
        str(float(config.get("learning_rate", 2e-5))),
        "--seed",
        str(int(config.get("seed", 42))),
        "--fine-tune-type",
        str(config.get("fine_tune_type", "lora")),
    ]

    if config.get("steps_per_report") is not None:
        command.extend(["--steps-per-report", str(int(config["steps_per_report"]))])
    if config.get("steps_per_eval") is not None:
        command.extend(["--steps-per-eval", str(int(config["steps_per_eval"]))])
    if config.get("save_every") is not None:
        command.extend(["--save-every", str(int(config["save_every"]))])
    if config.get("num_layers") is not None:
        command.extend(["--num-layers", str(int(config["num_layers"]))])
    if config.get("max_seq_length") is not None:
        command.extend(["--max-seq-length", str(int(config["max_seq_length"]))])

    extra_args = config.get("extra_args", [])
    if extra_args:
        command.extend([str(arg) for arg in extra_args])

    print("Running MLX LoRA training command:")
    print(" ".join(command))
    subprocess.run(command, check=True)

    manifest = {
        "pipeline": "mlx_lora",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "model_name": config["model_name"],
        "dataset_path": config["dataset_path"],
        "mlx_train_jsonl": train_jsonl,
        "adapter_path": str(adapter_path),
        "config": config,
    }
    manifest_path = adapter_path / "training_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    config_copy = adapter_path / "config_used.yaml"
    with open(config_copy, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return {
        "adapter_path": str(adapter_path),
        "manifest_path": str(manifest_path),
        "mlx_train_jsonl": train_jsonl,
    }
