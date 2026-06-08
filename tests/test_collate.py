from __future__ import annotations

import torch

from src.data.preprocess import collate_fn


def test_collate_fn_adds_attention_masks():
    batch = [
        {
            "input_ids_chosen": torch.tensor([1, 2, 3], dtype=torch.long),
            "input_ids_rejected": torch.tensor([1, 2], dtype=torch.long),
            "prompt_length": 2,
        },
        {
            "input_ids_chosen": torch.tensor([4], dtype=torch.long),
            "input_ids_rejected": torch.tensor([4, 5, 6], dtype=torch.long),
            "prompt_length": 1,
        },
    ]
    out = collate_fn(batch, pad_token_id=0)
    assert "attention_mask_chosen" in out
    assert "attention_mask_rejected" in out
    assert out["attention_mask_chosen"].shape == out["input_ids_chosen"].shape
    assert out["attention_mask_rejected"].shape == out["input_ids_rejected"].shape
