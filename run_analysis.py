from __future__ import annotations

import argparse
import json
import os

from transformers import AutoTokenizer

from src.data.analyze import compute_token_divergence, save_analysis
from src.data.download import download_ultrafeedback, load_pairs
from src.data.preprocess import format_for_dpo, save_formatted_dataset
from src.models.loader import load_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run DPO token-level data pipeline.")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--max-analysis-samples", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--analysis-device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device to use for token-divergence analysis forward passes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not (args.download or args.analyze or args.preprocess):
        raise SystemExit("Specify at least one action: --download, --analyze, --preprocess")

    if args.download:
        download_ultrafeedback(sample_size=args.sample_size)

    pairs = load_pairs()
    if not pairs:
        raise SystemExit("No raw pairs found. Run with --download first.")

    if args.analyze:
        model, tokenizer, _ = load_model_and_tokenizer(args.model_name, device=args.analysis_device)
        stats, per_pair = compute_token_divergence(
            model,
            tokenizer,
            pairs,
            threshold=args.threshold,
            max_samples=args.max_analysis_samples,
            max_length=args.max_length,
        )
        save_analysis(stats, per_pair, output_dir="results")
        with open("results/divergence_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    if args.preprocess:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        formatted = format_for_dpo(pairs, tokenizer, max_length=args.max_length)
        os.makedirs("data/processed", exist_ok=True)
        save_formatted_dataset(formatted, "data/processed/train.jsonl")


if __name__ == "__main__":
    main()
