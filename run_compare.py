from __future__ import annotations

import argparse
import json
import os
from typing import Any


def _load_result(path: str) -> tuple[str, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    model_name = data.get("model_name") or os.path.basename(path).replace("_results.json", "")
    metrics = data.get("metrics", data)
    return model_name, metrics


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def _fmt_num(value: float | None, suffix: str = "", decimals: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}{suffix}"


def _render_table(rows: list[list[str]]) -> str:
    header = "| Model | IF Accuracy | Refusal Accuracy | Consistency | TTFT (ms) | Tok/s |"
    sep = "|---|---:|---:|---:|---:|---:|"
    body = [
        f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |"
        for r in rows
    ]
    return "\n".join([header, sep, *body])


def parse_args():
    parser = argparse.ArgumentParser(description="Compare evaluation result JSON files.")
    parser.add_argument("--results", nargs="+", required=True, help="Paths to *_results.json files.")
    parser.add_argument("--output", default="results/comparison_table.md", help="Markdown output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for path in args.results:
        model_name, m = _load_result(path)
        rows.append(
            [
                model_name,
                _fmt_pct(m.get("instruction_following")),
                _fmt_pct(m.get("refusal_rate")),
                _fmt_num(m.get("consistency"), decimals=3),
                _fmt_num(m.get("mean_ttft_ms"), suffix="", decimals=1),
                _fmt_num(m.get("mean_throughput_toks_per_sec"), suffix="", decimals=1),
            ]
        )

    table = _render_table(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("# Model Comparison\n\n")
        f.write(table + "\n")
    print(table)
    print(f"\nSaved comparison table to {args.output}")


if __name__ == "__main__":
    main()
