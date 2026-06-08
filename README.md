# Where DPO Breaks: Token-Level Study + Apple-Silicon MLX Track

This repository analyzes token-level failure modes of DPO, implements a token-weighted DPO baseline in PyTorch, and adds an Apple-focused MLX+LoRA pipeline for constrained on-device hardware.

The project is now structured for reproducible public runs:
- deterministic seeds in training
- fail-fast config/data validation
- frozen eval suite with version manifest
- per-prompt auditable outputs (no hardcoded result numbers)
- common result schema across PyTorch and MLX evaluation tracks

## Why this repo is public-ready

- No static or hardcoded benchmark outcomes are embedded in code paths.
- Every reported metric is generated from runtime inference.
- Eval outputs include run metadata (`run_manifest*.json`) and per-prompt traces (`*_per_prompt.jsonl`).
- Pipeline supports baseline comparison:
  - base model
  - PyTorch DPO baseline
  - token-weighted DPO
  - MLX+LoRA adaptation

## Project structure

```text
dpo-token-level-study/
├── configs/
│   ├── dpo_baseline.yaml
│   ├── token_weighted_dpo.yaml
│   └── mlx_lora_dpo.yaml
├── evals/
│   ├── instruction_following.jsonl
│   ├── adversarial.jsonl
│   ├── consistency.jsonl
│   └── suite_manifest.json
├── scripts/
│   └── run_pipeline.py
├── src/
│   ├── data/
│   ├── eval/
│   └── training/
├── tests/
├── run_analysis.py
├── run_train.py
├── run_train_mlx.py
├── run_eval.py
├── run_eval_mlx.py
└── run_compare.py
```

## Hardware guidance (8GB M3 Air)

- **Preferred on-device adaptation:** MLX + LoRA (`run_train_mlx.py`).
- PyTorch DPO baseline remains available for research comparison, but two full-model DPO is memory-heavy.
- For constrained devices, keep:
  - `batch_size=1`
  - moderate `max_seq_length` (e.g., 256-512)
  - quantized MLX base models (e.g. `mlx-community/*-4bit`)

## Install

```bash
pip install -r requirements.txt
```

## Reproducible workflow

### 1) Data prep

```bash
python run_analysis.py --download --analyze --preprocess
```

### 2) Train PyTorch baselines

```bash
python run_train.py --config configs/dpo_baseline.yaml
python run_train.py --config configs/token_weighted_dpo.yaml
```

### 3) Evaluate PyTorch/base models

```bash
python run_eval.py \
  --models base=Qwen/Qwen3-1.7B \
           baseline=checkpoints/dpo_baseline/epoch_2 \
           token_weighted=checkpoints/token_weighted_dpo/epoch_2 \
  --suite evals \
  --output results/pytorch
```

### 4) Train and evaluate MLX+LoRA

```bash
python run_train_mlx.py --config configs/mlx_lora_dpo.yaml

python run_eval_mlx.py \
  --models mlx_base=mlx-community/Qwen3-1.7B-4bit \
           mlx_lora=mlx-community/Qwen3-1.7B-4bit@checkpoints/mlx_lora_dpo \
  --suite evals \
  --output results/mlx
```

### 5) Build comparison table

```bash
python run_compare.py \
  --results results/pytorch/base_results.json \
            results/pytorch/baseline_results.json \
            results/pytorch/token_weighted_results.json \
            results/mlx/mlx_base_results.json \
            results/mlx/mlx_lora_results.json \
  --output results/comparison_table.md
```

## One-command orchestrator

- Smoke run (quick validation):

```bash
python scripts/run_pipeline.py --mode smoke
```

- Full pipeline:

```bash
python scripts/run_pipeline.py --mode full
```

## Output artifacts

Each eval run emits:
- `*_results.json` (aggregate metrics + suite hash)
- `*_per_prompt.jsonl` (auditable per-prompt responses/scores)
- `run_manifest.json` or `run_manifest_mlx.json` (runtime metadata/provenance)
- `summary.md` and optional `comparison_table.md`

## Before vs After (Metal, same suite)

From `results/mlx_before_after_metal/comparison_table.md`, both runs used:
- model family: `mlx-community/Qwen3-1.7B-4bit`
- backend: MLX Metal (`Device(gpu, 0)` on Apple silicon)
- eval suite: `evals` (`suite_manifest v1.0.0`, 30 prompts total)
- generation length: `max_new_tokens=80`

| Model | IF Accuracy | Refusal Accuracy | Consistency | TTFT (ms) | Tok/s |
|---|---:|---:|---:|---:|---:|
| base_mlx | 0.0% | 87.5% | 0.555 | 333.1 | 25.1 |
| lora_mlx | 73.3% | 100.0% | 0.584 | 331.2 | 22.2 |

Interpretation:
- LoRA fine-tuning significantly improved instruction-following and refusal behavior on this frozen suite.
- Latency remained comparable on Metal (TTFT roughly unchanged, modest throughput drop).
- This table is generated from runtime outputs, not hardcoded values.

## Test and smoke coverage

Run tests:

```bash
python -m pytest tests -q
```

Current suite includes:
- DPO masking correctness tests
- collate/padding mask tests
- config validation tests
- eval scorer/schema sanity tests

## Notes

- Freeze `evals/*` once benchmarking begins; suite version is tracked via `suite_manifest.json` + content hash.
- W&B is optional in training (`use_wandb: true/false`).
- For apples-to-apples comparisons, keep eval suite and generation settings fixed across all model runs.
