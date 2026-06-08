from __future__ import annotations

import argparse
import os

import yaml

from src.training.mlx_lora_dpo import train_mlx_lora


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLX LoRA adapters for on-device pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to MLX LoRA YAML config.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.config):
        raise SystemExit(f"Config file not found: {args.config}")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise SystemExit("Config file must parse to a dictionary.")

    output = train_mlx_lora(config)
    print(f"MLX LoRA training completed. Adapter path: {output['adapter_path']}")


if __name__ == "__main__":
    main()
