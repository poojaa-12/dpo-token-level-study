from __future__ import annotations

import argparse

import yaml

from src.training.trainer import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train DPO model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train(config)


if __name__ == "__main__":
    main()
