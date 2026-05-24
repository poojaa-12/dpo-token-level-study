from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_torch_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def load_model_and_tokenizer(model_name: str, device: str | None = None):
    device = device or detect_device()
    dtype = resolve_torch_dtype(device)

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device
