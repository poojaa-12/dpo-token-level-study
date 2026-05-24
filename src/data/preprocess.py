from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence


def format_for_dpo(pairs: list[dict[str, Any]], tokenizer, max_length: int = 512):
    """
    Convert preference pairs into tokenized DPO format.
    """
    formatted = []
    for pair in pairs:
        prompt = f"### Instruction:\n{pair['prompt']}\n\n### Response:\n"
        chosen_full = prompt + pair["chosen"]
        rejected_full = prompt + pair["rejected"]

        chosen_enc = tokenizer(
            chosen_full,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        rejected_enc = tokenizer(
            rejected_full,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_len = len(tokenizer(prompt).input_ids)

        formatted.append(
            {
                "input_ids_chosen": chosen_enc["input_ids"].squeeze(0).tolist(),
                "input_ids_rejected": rejected_enc["input_ids"].squeeze(0).tolist(),
                "prompt_length": prompt_len,
            }
        )
    return formatted


def save_formatted_dataset(formatted: list[dict[str, Any]], path: str = "data/processed/train.jsonl") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in formatted:
            f.write(json.dumps(row) + "\n")
    print(f"Saved formatted dataset: {path} ({len(formatted)} rows)")
    return path


@dataclass
class DPODataset(torch.utils.data.Dataset):
    rows: list[dict[str, Any]]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            "input_ids_chosen": torch.tensor(row["input_ids_chosen"], dtype=torch.long),
            "input_ids_rejected": torch.tensor(row["input_ids_rejected"], dtype=torch.long),
            "prompt_length": int(row["prompt_length"]),
        }


def load_formatted_dataset(path: str = "data/processed/train.jsonl") -> DPODataset:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return DPODataset(rows)


def collate_fn(batch: list[dict[str, Any]], pad_token_id: int):
    chosen = [item["input_ids_chosen"] for item in batch]
    rejected = [item["input_ids_rejected"] for item in batch]
    prompt_lengths = [item["prompt_length"] for item in batch]

    chosen_padded = pad_sequence(chosen, batch_first=True, padding_value=pad_token_id)
    rejected_padded = pad_sequence(rejected, batch_first=True, padding_value=pad_token_id)
    prompt_length_tensor = torch.tensor(prompt_lengths, dtype=torch.long)

    return {
        "input_ids_chosen": chosen_padded,
        "input_ids_rejected": rejected_padded,
        "prompt_length": prompt_length_tensor,
    }
