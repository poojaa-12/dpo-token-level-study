from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run(command: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end DPO pipeline orchestrator.")
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help="Smoke mode runs quick checks; full mode runs the complete default pipeline.",
    )
    parser.add_argument("--config", default="configs/dpo_baseline.yaml", help="PyTorch train config.")
    parser.add_argument("--mlx-config", default="configs/mlx_lora_dpo.yaml", help="MLX train config.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-mlx-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if args.mode == "smoke":
        _run(
            [
                "python3",
                "run_analysis.py",
                "--preprocess",
                "--max-length",
                "256",
            ],
            repo_root,
        )
        if not args.skip_train:
            _run(["python3", "run_train.py", "--config", args.config], repo_root)
        if not args.skip_eval:
            _run(
                [
                    "python3",
                    "run_eval.py",
                    "--models",
                    "base=Qwen/Qwen3-1.7B",
                    "baseline=checkpoints/dpo_baseline/epoch_0",
                    "--output",
                    "results/smoke",
                    "--max-new-tokens",
                    "64",
                ],
                repo_root,
            )
    else:
        _run(["python3", "run_analysis.py", "--download", "--analyze", "--preprocess"], repo_root)
        if not args.skip_train:
            _run(["python3", "run_train.py", "--config", args.config], repo_root)
        if not args.skip_mlx_train:
            _run(["python3", "run_train_mlx.py", "--config", args.mlx_config], repo_root)
        if not args.skip_eval:
            _run(
                [
                    "python3",
                    "run_eval.py",
                    "--models",
                    "base=Qwen/Qwen3-1.7B",
                    "baseline=checkpoints/dpo_baseline/epoch_2",
                    "token_weighted=checkpoints/token_weighted_dpo/epoch_2",
                    "--output",
                    "results/full",
                ],
                repo_root,
            )
            _run(
                [
                    "python3",
                    "run_eval_mlx.py",
                    "--models",
                    "mlx_base=mlx-community/Qwen3-1.7B-4bit",
                    "mlx_lora=mlx-community/Qwen3-1.7B-4bit@checkpoints/mlx_lora_dpo",
                    "--output",
                    "results/full",
                ],
                repo_root,
            )
            _run(
                [
                    "python3",
                    "run_compare.py",
                    "--results",
                    "results/full/base_results.json",
                    "results/full/baseline_results.json",
                    "results/full/token_weighted_results.json",
                    "results/full/mlx_base_results.json",
                    "results/full/mlx_lora_results.json",
                    "--output",
                    "results/full/comparison_table.md",
                ],
                repo_root,
            )


if __name__ == "__main__":
    main()
