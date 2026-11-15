# GraphT5

This repository contains the model, data loader, and train/evaluate entry points for the GraphT5. 

The model has a CodeT5 passage encoder, a six-layer edge-masked VDG encoder, and a decoder with gated text and graph cross-attention.

## Layout

- `src/grapht5/modeling/`: architecture.
- `src/grapht5/datasets/`: serialized sample schema, packing, and batching.
- `src/grapht5/pipeline/`, `src/grapht5/cli/`: training and inference.
- `scripts/`: training and evaluation entry points.
- `configs/*.toml`: paper pretraining and dataset-specific fine-tuning.
- `datasets/`: prepared pretraining and fine-tuning splits, benchmark definitions, and file checksums.
- `tests/`: architecture and input-contract checks.
- `generation/`: diff/tokenization, passage preparation, and Joern VDG construction source.

Install with `uv sync --extra dev`, then run `uv run pytest`. The scripts take prepared dataset and checkpoint paths as positional arguments:

```bash
bash scripts/pretrain.sh TRAIN.jsonl VALID.jsonl CHECKPOINT_DIR
bash scripts/finetune.sh TRAIN.jsonl DEV.jsonl PRETRAINED_CHECKPOINT CHECKPOINT_DIR bigvul_cvefixes
bash scripts/evaluate.sh TEST.jsonl BEST_CHECKPOINT OUTPUT_DIR bigvul_cvefixes 10
```

Dataset names are `bigvul_cvefixes`, `icvul`, `primevul`, `vul4j`, and `vul4c`. Set `GPUS` for training if needed. The evaluation script's final argument is the candidate count. Each TOML file inherits shared settings from `configs/base.toml`.

Prepared datasets are included in `datasets/`.
