from __future__ import annotations

import torch
import torch.nn.functional as F

from .dpo_loss import dpo_loss


def compute_token_weights(
    reference_model,
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
    prompt_length: torch.Tensor,
    chosen_attention_mask: torch.Tensor | None = None,
    rejected_attention_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    min_weight: float = 0.1,
):
    if chosen_ids.shape[0] != 1 or rejected_ids.shape[0] != 1:
        raise ValueError("Token-weighted DPO currently supports batch_size=1 only.")

    prompt_start = max(int(prompt_length[0].item()) - 1, 0)

    with torch.no_grad():
        chosen_logits = reference_model(chosen_ids).logits
        rejected_logits = reference_model(rejected_ids).logits

    chosen_lp = F.log_softmax(chosen_logits, dim=-1)
    rejected_lp = F.log_softmax(rejected_logits, dim=-1)

    chosen_valid = int(chosen_attention_mask[0].sum().item()) if chosen_attention_mask is not None else chosen_ids.shape[1]
    rejected_valid = (
        int(rejected_attention_mask[0].sum().item())
        if rejected_attention_mask is not None
        else rejected_ids.shape[1]
    )
    min_len = min(chosen_valid, rejected_valid)
    weights = []
    for pos in range(prompt_start, min_len - 1):
        c_tok = chosen_ids[0, pos + 1].item()
        r_tok = rejected_ids[0, pos + 1].item()
        divergence = abs(chosen_lp[0, pos, c_tok].item() - rejected_lp[0, pos, r_tok].item())
        weights.append(divergence)

    if not weights:
        return None

    w = torch.tensor(weights, dtype=torch.float32, device=chosen_ids.device)
    w = F.softmax(w / temperature, dim=0)
    w = w.clamp(min=min_weight)
    w = w / w.sum()
    return w


def token_weighted_log_probs(model, input_ids: torch.Tensor, prompt_length: torch.Tensor, token_weights: torch.Tensor):
    logits = model(input_ids).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    response_start = max(int(prompt_length[0].item()) - 1, 0)
    response_end = response_start + len(token_weights)
    response_lp = token_log_probs[0, response_start:response_end]
    return (response_lp * token_weights.to(response_lp.device)).sum()


def token_weighted_dpo_loss(
    policy_model,
    reference_model,
    batch,
    beta: float = 0.1,
    weight_temperature: float = 1.0,
):
    chosen_ids = batch["input_ids_chosen"]
    rejected_ids = batch["input_ids_rejected"]
    chosen_attention = batch.get("attention_mask_chosen")
    rejected_attention = batch.get("attention_mask_rejected")
    prompt_len = batch["prompt_length"]

    token_weights = compute_token_weights(
        reference_model,
        chosen_ids,
        rejected_ids,
        prompt_len,
        chosen_attention_mask=chosen_attention,
        rejected_attention_mask=rejected_attention,
        temperature=weight_temperature,
    )

    if token_weights is None:
        return dpo_loss(policy_model, reference_model, batch, beta=beta)

    policy_chosen_lp = token_weighted_log_probs(policy_model, chosen_ids, prompt_len, token_weights)
    policy_rejected_lp = token_weighted_log_probs(policy_model, rejected_ids, prompt_len, token_weights)

    with torch.no_grad():
        ref_chosen_lp = token_weighted_log_probs(reference_model, chosen_ids, prompt_len, token_weights)
        ref_rejected_lp = token_weighted_log_probs(reference_model, rejected_ids, prompt_len, token_weights)

    chosen_reward = beta * (policy_chosen_lp - ref_chosen_lp)
    rejected_reward = beta * (policy_rejected_lp - ref_rejected_lp)
    loss = -F.logsigmoid(chosen_reward - rejected_reward)

    return loss, {
        "loss": float(loss.item()),
        "reward_margin": float((chosen_reward - rejected_reward).item()),
        "mean_token_weight_entropy": float(-(token_weights * token_weights.log()).sum().item()),
    }
