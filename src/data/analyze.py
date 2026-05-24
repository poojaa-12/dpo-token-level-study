from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


def compute_token_divergence(
    model,
    tokenizer,
    pairs: list[dict[str, Any]],
    threshold: float = 0.5,
    max_samples: int = 500,
    max_length: int = 512,
):
    """
    Compute token-level chosen vs rejected log-prob divergence.
    """
    model.eval()
    per_pair_data: list[dict[str, Any]] = []
    device = next(model.parameters()).device

    skipped = 0
    for pair in tqdm(pairs[:max_samples], desc="Analyzing token divergence"):
        prompt = pair["prompt"]
        chosen = pair["chosen"]
        rejected = pair["rejected"]

        chosen_ids = tokenizer(
            prompt + chosen,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).input_ids.to(device)
        rejected_ids = tokenizer(
            prompt + rejected,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).input_ids.to(device)
        prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)

        try:
            with torch.no_grad():
                chosen_logits = model(chosen_ids).logits[0]
                rejected_logits = model(rejected_ids).logits[0]
        except RuntimeError as e:
            skipped += 1
            print(f"Skipping sample due to model forward error: {e}")
            continue

        chosen_logprobs = torch.log_softmax(chosen_logits, dim=-1)
        rejected_logprobs = torch.log_softmax(rejected_logits, dim=-1)

        min_len = min(chosen_ids.shape[1], rejected_ids.shape[1])
        overlap_len = min_len - prompt_len
        if overlap_len <= 1:
            continue

        chosen_token_lp = []
        rejected_token_lp = []

        for pos in range(prompt_len, min_len - 1):
            c_tok = chosen_ids[0, pos + 1].item()
            r_tok = rejected_ids[0, pos + 1].item()
            chosen_token_lp.append(chosen_logprobs[pos, c_tok].item())
            rejected_token_lp.append(rejected_logprobs[pos, r_tok].item())

        divergence = np.array(chosen_token_lp) - np.array(rejected_token_lp)
        mask = np.abs(divergence) > threshold
        first_div = int(np.argmax(mask)) if mask.any() else -1

        per_pair_data.append(
            {
                "divergence": divergence.tolist(),
                "chosen_better_frac": float((divergence > threshold).mean()),
                "rejected_better_frac": float((divergence < -threshold).mean()),
                "neutral_frac": float((np.abs(divergence) <= threshold).mean()),
                "first_divergence_pos": first_div,
            }
        )

    if not per_pair_data:
        raise ValueError("No valid pairs found for divergence analysis.")

    chosen_better = float(np.mean([d["chosen_better_frac"] for d in per_pair_data]))
    rejected_better = float(np.mean([d["rejected_better_frac"] for d in per_pair_data]))
    neutral = float(np.mean([d["neutral_frac"] for d in per_pair_data]))

    first_positions = [d["first_divergence_pos"] for d in per_pair_data if d["first_divergence_pos"] >= 0]
    mean_first_divergence = float(np.mean(first_positions)) if first_positions else -1.0

    divergence_stats = {
        "chosen_better_fraction": chosen_better,
        "rejected_better_fraction": rejected_better,
        "neutral_fraction": neutral,
        "mean_first_divergence_position": mean_first_divergence,
    }

    print("\n=== Token-level Divergence Analysis ===")
    print(f"Chosen tokens genuinely better:   {chosen_better:.1%}")
    print(f"Rejected tokens genuinely better: {rejected_better:.1%}")
    print(f"Neutral (indistinguishable):      {neutral:.1%}")
    if skipped:
        print(f"Skipped pairs due to runtime issues: {skipped}")
    print(f"\nThis means DPO is training on noisy signal for {neutral:.1%} of tokens.")

    return divergence_stats, per_pair_data


def save_analysis(
    divergence_stats: dict[str, Any],
    per_pair_data: list[dict[str, Any]],
    output_dir: str = "results",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "token_divergence_analysis.json")
    payload = {
        "aggregate": divergence_stats,
        "samples_analyzed": len(per_pair_data),
        "per_pair": per_pair_data,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved analysis to {output_path}")
    return output_path
