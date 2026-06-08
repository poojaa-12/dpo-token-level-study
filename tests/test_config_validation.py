from __future__ import annotations

import json

import pytest

from src.training.trainer import _validate_train_config


def test_validate_train_config_missing_dataset(tmp_path):
    cfg = {
        "model_name": "Qwen/Qwen3-1.7B",
        "batch_size": 1,
        "lr": 1e-5,
        "save_path": "checkpoints/test",
        "dataset_path": str(tmp_path / "missing.jsonl"),
    }
    with pytest.raises(FileNotFoundError):
        _validate_train_config(cfg)


def test_validate_train_config_token_weighted_requires_batch_one(tmp_path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(json.dumps({"x": 1}) + "\n", encoding="utf-8")
    cfg = {
        "model_name": "Qwen/Qwen3-1.7B",
        "batch_size": 2,
        "lr": 1e-5,
        "save_path": "checkpoints/test",
        "dataset_path": str(dataset),
        "use_token_weighted": True,
    }
    with pytest.raises(ValueError):
        _validate_train_config(cfg)
