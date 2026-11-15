#!/usr/bin/env bash

train_data="$1"
valid_data="$2"
pretrained_checkpoint="$3"
checkpoint_dir="$4"
dataset="$5"
gpus="${GPUS:-1}"
package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="$package_root/configs/finetuning_${dataset}.toml"

uv run --frozen --project "$package_root" torchrun --nproc_per_node "$gpus" -m grapht5.cli.train \
  --config "$config" \
  --train-data "$train_data" --eval-data "$valid_data" \
  --model-path "$pretrained_checkpoint" --checkpoint-dir "$checkpoint_dir"
