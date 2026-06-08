from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from collections import defaultdict
from typing import Any

from src.eval.report import write_jsonl, write_results_json, write_run_manifest, write_summary
from src.eval.scorer import score_consistency, score_instruction_following, score_refusal
from src.eval.suite import compute_suite_hash, load_eval_suite


def parse_model_args(models: list[str]) -> dict[str, dict[str, str]]:
    """
    Parse model specs like:
      name=model_path
      name=model_path@adapter_path
    """
    parsed = {}
    for item in models:
        if "=" not in item:
            raise ValueError(f"Invalid --models entry: {item}. Expected name=model or name=model@adapter")
        name, spec = item.split("=", 1)
        model_path, adapter_path = (spec.split("@", 1) + [""])[:2]
        parsed[name.strip()] = {"model_path": model_path.strip(), "adapter_path": adapter_path.strip() or None}
    return parsed


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


def _generate_response(model, tokenizer, prompt: str, max_tokens: int = 128) -> tuple[str, float, float | None]:
    import mlx_lm

    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted = prompt

    start = time.perf_counter()
    first_token_time = None
    output_text = []
    token_count = 0
    for chunk in mlx_lm.stream_generate(model, tokenizer, formatted, max_tokens=max_tokens):
        token_count += 1
        if first_token_time is None:
            first_token_time = (time.perf_counter() - start) * 1000
        text_piece = getattr(chunk, "text", "")
        if text_piece:
            output_text.append(text_piece)
    elapsed = max(time.perf_counter() - start, 1e-9)
    response = "".join(output_text).strip()
    throughput = token_count / elapsed if token_count > 0 else None
    ttft = first_token_time if first_token_time is not None else elapsed * 1000
    return response, ttft, throughput


def evaluate_model(spec: dict[str, str], suite: dict, max_new_tokens: int = 128):
    import mlx_lm

    model, tokenizer = mlx_lm.load(spec["model_path"], adapter_path=spec.get("adapter_path"))

    inst_scores = []
    refusal_scores = []
    grouped = defaultdict(list)
    prompt_rows = []
    consistency_rows = []
    ttfts = []
    throughputs = []

    for row in suite["instruction_following"]:
        response, ttft, throughput = _generate_response(model, tokenizer, row["prompt"], max_tokens=max_new_tokens)
        constraints = _extract_constraints(row)
        score = score_instruction_following(
            response=response,
            constraints=constraints,
            expected_format=row.get("expected_format", ""),
            expected_count=row.get("expected_count"),
        )
        inst_scores.append(score)
        ttfts.append(ttft)
        if throughput is not None:
            throughputs.append(throughput)
        prompt_rows.append(
            {
                "id": row.get("id"),
                "category": "instruction_following",
                "prompt": row["prompt"],
                "response": response,
                "constraints": constraints,
                "score": score,
                "ttft_ms": round(ttft, 1),
                "throughput_toks_per_sec": round(throughput, 1) if throughput is not None else None,
            }
        )

    for row in suite["adversarial"]:
        response, ttft, throughput = _generate_response(model, tokenizer, row["prompt"], max_tokens=max_new_tokens)
        score = score_refusal(response, bool(row.get("should_refuse", False)))
        refusal_scores.append(score)
        ttfts.append(ttft)
        if throughput is not None:
            throughputs.append(throughput)
        prompt_rows.append(
            {
                "id": row.get("id"),
                "category": "adversarial",
                "prompt": row["prompt"],
                "response": response,
                "should_refuse": bool(row.get("should_refuse", False)),
                "score": score,
                "ttft_ms": round(ttft, 1),
                "throughput_toks_per_sec": round(throughput, 1) if throughput is not None else None,
            }
        )

    grouped_rows = defaultdict(list)
    for row in suite["consistency"]:
        response, ttft, throughput = _generate_response(model, tokenizer, row["prompt"], max_tokens=max_new_tokens)
        grouped[row["group"]].append(response)
        grouped_rows[row["group"]].append(
            {
                "id": row.get("id"),
                "category": "consistency",
                "group": row["group"],
                "prompt": row["prompt"],
                "response": response,
                "ttft_ms": round(ttft, 1),
                "throughput_toks_per_sec": round(throughput, 1) if throughput is not None else None,
            }
        )
        ttfts.append(ttft)
        if throughput is not None:
            throughputs.append(throughput)

    consistency = score_consistency(dict(grouped))
    for group, rows in grouped_rows.items():
        group_score = score_consistency({group: [r["response"] for r in rows]})
        for row in rows:
            row["consistency_score"] = group_score
            consistency_rows.append(row)

    metrics = {
        "instruction_following": sum(inst_scores) / len(inst_scores) if inst_scores else 0.0,
        "refusal_rate": sum(refusal_scores) / len(refusal_scores) if refusal_scores else 0.0,
        "consistency": consistency,
        "mean_ttft_ms": sum(ttfts) / len(ttfts) if ttfts else None,
        "mean_throughput_toks_per_sec": sum(throughputs) / len(throughputs) if throughputs else None,
    }
    return metrics, prompt_rows, consistency_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MLX models/adapters with the common result schema.")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Model specs in form name=model_path or name=model_path@adapter_path",
    )
    parser.add_argument("--suite", type=str, default="evals")
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.suite):
        raise SystemExit(f"Eval suite directory not found: {args.suite}")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be > 0")

    try:
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "mlx_lm is not installed. Install with: pip install 'mlx-lm[train]' mlx"
        ) from exc

    models = parse_model_args(args.models)
    suite = load_eval_suite(args.suite)
    suite_hash = compute_suite_hash(args.suite)
    suite_meta_path = os.path.join(args.suite, "suite_manifest.json")
    if os.path.exists(suite_meta_path):
        with open(suite_meta_path, "r", encoding="utf-8") as f:
            suite_manifest = json.load(f)
    else:
        suite_manifest = {"version": "unversioned"}

    results_by_model = {}
    for name, spec in models.items():
        print(f"Evaluating {name} ({spec['model_path']})...")
        metrics, per_prompt_rows, consistency_rows = evaluate_model(
            spec,
            suite,
            max_new_tokens=args.max_new_tokens,
        )
        results_by_model[name] = metrics
        result_payload = {
            "model_name": name,
            "model_path": spec["model_path"],
            "adapter_path": spec.get("adapter_path"),
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
            "runtime": "mlx",
            "models": models,
            "suite_dir": args.suite,
            "suite_hash": suite_hash,
            "suite_manifest": suite_manifest,
            "max_new_tokens": args.max_new_tokens,
        },
        args.output,
        filename="run_manifest_mlx.json",
    )

    summary_path = write_summary(results_by_model, args.output)
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
