from __future__ import annotations

import os

import torch
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.preprocess import collate_fn, load_formatted_dataset
from .dpo_loss import dpo_loss
from .token_weighted_dpo import token_weighted_dpo_loss


def _resolve_dtype(device: str):
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def train(config: dict):
    model_name = config["model_name"]
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = _resolve_dtype(device)

    print(f"Loading {model_name} on {device}...")
    if device == "cuda":
        policy = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
        reference = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
    else:
        policy = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        reference = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)

    for param in reference.parameters():
        param.requires_grad = False
    reference.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_formatted_dataset(config.get("dataset_path", "data/processed/train.jsonl"))
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )

    optimizer = AdamW(policy.parameters(), lr=config["lr"], weight_decay=0.01)
    use_token_weighted = bool(config.get("use_token_weighted", False))
    weight_temperature = float(config.get("weight_temperature", 1.0))
    grad_accum_steps = int(config.get("grad_accum_steps", 1))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    beta = float(config.get("beta", 0.1))
    epochs = int(config.get("epochs", 3))
    save_path = config["save_path"]

    os.makedirs(save_path, exist_ok=True)

    wandb.init(project="dpo-token-level-study", config=config)

    step = 0
    for epoch in range(epochs):
        policy.train()
        optimizer.zero_grad()
        last_metrics = {"loss": 0.0}

        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            if use_token_weighted:
                loss, metrics = token_weighted_dpo_loss(
                    policy,
                    reference,
                    {
                        "input_ids_chosen": batch["input_ids_chosen"][:1],
                        "input_ids_rejected": batch["input_ids_rejected"][:1],
                        "prompt_length": batch["prompt_length"][:1],
                    },
                    beta=beta,
                    weight_temperature=weight_temperature,
                )
            else:
                loss, metrics = dpo_loss(policy, reference, batch, beta=beta)

            (loss / grad_accum_steps).backward()
            last_metrics = metrics

            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                wandb.log({**metrics, "epoch": epoch, "step": step})
                step += 1

        epoch_dir = os.path.join(save_path, f"epoch_{epoch}")
        policy.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        print(f"Epoch {epoch} complete. Loss: {last_metrics['loss']:.4f}")

    wandb.finish()
    return policy
