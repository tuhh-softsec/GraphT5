#!/usr/bin/env bash

tool="${1:?Use diff, vdg, or vul4c}"
shift

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
export PYTHONPATH="$root/generation/source"
cd "${GRAPHT5_GENERATION_WORKDIR:-$scratch}"

case "$tool" in
  diff) uv run --project "$root/generation" python -m code_graph.data_utils.mod.gen_tags "$@" ;;
  vdg) uv run --project "$root/generation" python "$root/generation/generate_vdg.py" "$@" ;;
  vul4c) uv run --project "$root/generation" python "$root/generation/source/data/grapht5/vul4c/convert_vul4c_to_grapht5.py" "$@" ;;
  *) exit 2 ;;
esac
