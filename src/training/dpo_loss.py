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


def compute_log_probs(model, input_ids: torch.Tensor, prompt_length: torch.Tensor):
    """
    Sum log-probs over response tokens only.
    """
    token_log_probs = _sequence_log_probs(model, input_ids)
    batch_size, seq_minus_one = token_log_probs.shape
    arange = torch.arange(seq_minus_one, device=token_log_probs.device).unsqueeze(0)
    mask = (arange >= prompt_length.unsqueeze(1)).float()
    return (token_log_probs * mask).sum(dim=-1)


def dpo_loss(policy_model, reference_model, batch, beta: float = 0.1):
    chosen_ids = batch["input_ids_chosen"]
    rejected_ids = batch["input_ids_rejected"]
    prompt_len = batch["prompt_length"]

    policy_chosen_lp = compute_log_probs(policy_model, chosen_ids, prompt_len)
    policy_rejected_lp = compute_log_probs(policy_model, rejected_ids, prompt_len)

    with torch.no_grad():
        ref_chosen_lp = compute_log_probs(reference_model, chosen_ids, prompt_len)
        ref_rejected_lp = compute_log_probs(reference_model, rejected_ids, prompt_len)

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
