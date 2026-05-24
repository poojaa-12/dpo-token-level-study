from __future__ import annotations

import argparse
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval.report import write_results_json, write_summary
from src.eval.scorer import score_consistency, score_instruction_following, score_refusal
from src.eval.suite import load_eval_suite


def parse_model_args(models: list[str]) -> dict[str, str]:
    parsed = {}
    for item in models:
        if "=" not in item:
            raise ValueError(f"Invalid --models entry: {item}. Expected name=path")
        name, path = item.split("=", 1)
        parsed[name.strip()] = path.strip()
    return parsed


def generate_response(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()


def evaluate_model(model_path: str, suite: dict) -> dict[str, float]:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    inst_scores = []
    for row in suite["instruction_following"]:
        response = generate_response(model, tokenizer, row["prompt"], device)
        inst_scores.append(
            score_instruction_following(
                response=response,
                expected_format=row.get("expected_format", ""),
                expected_count=row.get("expected_count"),
            )
        )

    refusal_scores = []
    for row in suite["adversarial"]:
        response = generate_response(model, tokenizer, row["prompt"], device)
        refusal_scores.append(score_refusal(response, bool(row.get("should_refuse", False))))

    grouped = defaultdict(list)
    for row in suite["consistency"]:
        response = generate_response(model, tokenizer, row["prompt"], device)
        grouped[row["group"]].append(response)
    consistency = score_consistency(dict(grouped))

    return {
        "instruction_following": sum(inst_scores) / len(inst_scores) if inst_scores else 0.0,
        "refusal_rate": sum(refusal_scores) / len(refusal_scores) if refusal_scores else 0.0,
        "consistency": consistency,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate base and fine-tuned models.")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Model specs in form name=hf_or_local_path",
    )
    parser.add_argument("--suite", type=str, default="evals")
    parser.add_argument("--output", type=str, default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    models = parse_model_args(args.models)
    suite = load_eval_suite(args.suite)

    results_by_model = {}
    for name, path in models.items():
        print(f"Evaluating {name} ({path})...")
        metrics = evaluate_model(path, suite)
        results_by_model[name] = metrics
        write_results_json(metrics, args.output, f"{name}_results.json")

    summary_path = write_summary(results_by_model, args.output)
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
