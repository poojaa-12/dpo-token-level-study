from __future__ import annotations

import torch
import torch.nn.functional as F


def _sequence_log_probs(model, input_ids: torch.Tensor):
    logits = model(input_ids).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs


def _response_mask(prompt_length: torch.Tensor, seq_minus_one: int, device: torch.device) -> torch.Tensor:
    # token_log_probs[i] predicts token at index i+1, so first response token starts at prompt_len-1
    start = (prompt_length - 1).clamp(min=0).unsqueeze(1)
    positions = torch.arange(seq_minus_one, device=device).unsqueeze(0)
    return positions >= start


def compute_log_probs(
    model,
    input_ids: torch.Tensor,
    prompt_length: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
):
    """
    Sum log-probs over response tokens only.
    """
    token_log_probs = _sequence_log_probs(model, input_ids)
    batch_size, seq_minus_one = token_log_probs.shape
    response_mask = _response_mask(prompt_length, seq_minus_one, token_log_probs.device)
    if attention_mask is not None:
        valid_mask = attention_mask[:, 1:].bool()
        response_mask = response_mask & valid_mask
    mask = response_mask.float()
    return (token_log_probs * mask).sum(dim=-1)


def dpo_loss(policy_model, reference_model, batch, beta: float = 0.1):
    chosen_ids = batch["input_ids_chosen"]
    rejected_ids = batch["input_ids_rejected"]
    chosen_attention = batch.get("attention_mask_chosen")
    rejected_attention = batch.get("attention_mask_rejected")
    prompt_len = batch["prompt_length"]

    policy_chosen_lp = compute_log_probs(policy_model, chosen_ids, prompt_len, chosen_attention)
    policy_rejected_lp = compute_log_probs(policy_model, rejected_ids, prompt_len, rejected_attention)

    with torch.no_grad():
        ref_chosen_lp = compute_log_probs(reference_model, chosen_ids, prompt_len, chosen_attention)
        ref_rejected_lp = compute_log_probs(reference_model, rejected_ids, prompt_len, rejected_attention)

    chosen_reward = beta * (policy_chosen_lp - ref_chosen_lp)
    rejected_reward = beta * (policy_rejected_lp - ref_rejected_lp)

    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()

    with torch.no_grad():
        reward_margin = (chosen_reward - rejected_reward).mean()
        chosen_reward_mean = chosen_reward.mean()
        rejected_reward_mean = rejected_reward.mean()

    return loss, {
        "loss": float(loss.item()),
        "reward_margin": float(reward_margin.item()),
        "chosen_reward": float(chosen_reward_mean.item()),
        "rejected_reward": float(rejected_reward_mean.item()),
    }
