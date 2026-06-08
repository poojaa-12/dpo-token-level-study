from __future__ import annotations

import torch

from src.training.dpo_loss import _response_mask


def test_response_mask_includes_first_response_token():
    # prompt_length=4 means token at index 3 is first response token prediction position
    prompt_length = torch.tensor([4])
    mask = _response_mask(prompt_length, seq_minus_one=8, device=torch.device("cpu"))
    expected = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.bool)
    assert torch.equal(mask, expected)


def test_attention_mask_excludes_padding_positions():
    prompt_length = torch.tensor([2])
    base_mask = _response_mask(prompt_length, seq_minus_one=5, device=torch.device("cpu"))
    attention_shifted = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    masked = base_mask & attention_shifted
    expected = torch.tensor([[0, 1, 1, 0, 0]], dtype=torch.bool)
    assert torch.equal(masked, expected)
