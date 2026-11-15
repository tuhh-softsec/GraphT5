#!/usr/bin/env bash

train_data="$1"
valid_data="$2"
checkpoint_dir="$3"
gpus="${GPUS:-2}"
package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uv run --frozen --project "$package_root" torchrun --nproc_per_node "$gpus" -m grapht5.cli.train \
  --config "$package_root/configs/pretraining.toml" \
  --train-data "$train_data" --eval-data "$valid_data" \
  --checkpoint-dir "$checkpoint_dir"
