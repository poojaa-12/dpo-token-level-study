from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import platform
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.preprocess import format_instruction_prompt
from src.eval.report import write_jsonl, write_results_json, write_run_manifest, write_summary
from src.eval.scorer import score_consistency, score_instruction_following, score_refusal
from src.eval.suite import compute_suite_hash, load_eval_suite


def parse_model_args(models: list[str]) -> dict[str, str]:
    parsed = {}
    for item in models:
        if "=" not in item:
            raise ValueError(f"Invalid --models entry: {item}. Expected name=path")
        name, path = item.split("=", 1)
        parsed[name.strip()] = path.strip()
    return parsed


def generate_response(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 128) -> str:
    wrapped_prompt = format_instruction_prompt(prompt)
    inputs = tokenizer(wrapped_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_only = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_only, skip_special_tokens=True).strip()


def _get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _extract_constraints(row: dict) -> dict:
    if "constraints" in row and isinstance(row["constraints"], dict):
        return row["constraints"]
    constraints = {}
    if row.get("expected_format"):
        constraints["format"] = row["expected_format"]
    if row.get("expected_count") is not None:
        constraints["count"] = row["expected_count"]
    return constraints


def evaluate_model(
    model_path: str,
    suite: dict,
    device: str = "auto",
    max_new_tokens: int = 128,
) -> tuple[dict[str, float], list[dict], list[dict]]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    inst_scores = []
    prompt_rows = []
    consistency_prompt_rows = []
    for row in suite["instruction_following"]:
        response = generate_response(model, tokenizer, row["prompt"], device, max_new_tokens=max_new_tokens)
        constraints = _extract_constraints(row)
        score = score_instruction_following(
            response=response,
            constraints=constraints,
            expected_format=row.get("expected_format", ""),
            expected_count=row.get("expected_count"),
        )
        inst_scores.append(
            score
        )
        prompt_rows.append(
            {
                "id": row.get("id"),
                "category": "instruction_following",
                "prompt": row["prompt"],
                "response": response,
                "constraints": constraints,
                "score": score,
            }
        )

    refusal_scores = []
    for row in suite["adversarial"]:
        response = generate_response(model, tokenizer, row["prompt"], device, max_new_tokens=max_new_tokens)
        score = score_refusal(response, bool(row.get("should_refuse", False)))
        refusal_scores.append(score)
        prompt_rows.append(
            {
                "id": row.get("id"),
                "category": "adversarial",
                "prompt": row["prompt"],
                "response": response,
                "should_refuse": bool(row.get("should_refuse", False)),
                "score": score,
            }
        )

    grouped = defaultdict(list)
    grouped_rows = defaultdict(list)
    for row in suite["consistency"]:
        response = generate_response(model, tokenizer, row["prompt"], device, max_new_tokens=max_new_tokens)
        grouped[row["group"]].append(response)
        grouped_rows[row["group"]].append(
            {
                "id": row.get("id"),
                "category": "consistency",
                "group": row["group"],
                "prompt": row["prompt"],
                "response": response,
            }
        )
    consistency = score_consistency(dict(grouped))
    for group, rows in grouped_rows.items():
        group_score = score_consistency({group: [r["response"] for r in rows]})
        for row in rows:
            row["consistency_score"] = group_score
            consistency_prompt_rows.append(row)

    metrics = {
        "instruction_following": sum(inst_scores) / len(inst_scores) if inst_scores else 0.0,
        "refusal_rate": sum(refusal_scores) / len(refusal_scores) if refusal_scores else 0.0,
        "consistency": consistency,
    }
    return metrics, prompt_rows, consistency_prompt_rows


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
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.suite):
        raise SystemExit(f"Eval suite directory not found: {args.suite}")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be > 0")

    models = parse_model_args(args.models)
    for name, path in models.items():
        if not path:
            raise SystemExit(f"Empty model path for '{name}'")
    suite = load_eval_suite(args.suite)
    suite_hash = compute_suite_hash(args.suite)
    suite_meta_path = os.path.join(args.suite, "suite_manifest.json")
    if os.path.exists(suite_meta_path):
        with open(suite_meta_path, "r", encoding="utf-8") as f:
            suite_manifest = json.load(f)
    else:
        suite_manifest = {"version": "unversioned"}

    results_by_model = {}
    for name, path in models.items():
        print(f"Evaluating {name} ({path})...")
        metrics, per_prompt_rows, consistency_rows = evaluate_model(
            path,
            suite,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        results_by_model[name] = metrics
        result_payload = {
            "model_name": name,
            "model_path": path,
            "metrics": metrics,
            "suite_hash": suite_hash,
            "suite_manifest": suite_manifest,
            "max_new_tokens": args.max_new_tokens,
        }
        write_results_json(result_payload, args.output, f"{name}_results.json")
        write_jsonl(per_prompt_rows + consistency_rows, args.output, f"{name}_per_prompt.jsonl")

    write_run_manifest(
        {
            "git_sha": _get_git_sha(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "device": args.device,
            "models": models,
            "suite_dir": args.suite,
            "suite_hash": suite_hash,
            "suite_manifest": suite_manifest,
            "max_new_tokens": args.max_new_tokens,
        },
        args.output,
    )

    summary_path = write_summary(results_by_model, args.output)
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
