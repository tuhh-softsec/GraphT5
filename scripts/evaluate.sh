#!/usr/bin/env bash

test_data="$1"
checkpoint="$2"
output_dir="$3"
dataset="$4"
candidates="${5:-10}"
package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="$package_root/configs/finetuning_${dataset}.toml"

uv run --frozen --project "$package_root" python -m grapht5.cli.eval \
  --config "$config" --eval-data "$test_data" --model-path "$checkpoint" \
  --checkpoint-dir "$output_dir" --name "${dataset}_evaluation" \
  --beam-size "$candidates" --num-return-sequences "$candidates" \
  --write-results
