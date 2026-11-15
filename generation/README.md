# Dataset preparation source

`source/` contains the dataset diff/tokenization and Joern VDG construction code. `generate_vdg.py` is a small entry point for generating and attaching VDGs to an intermediate dataset.

Install the separate generation environment with `uv sync --project generation`. Joern and `clang-format` must also be installed and available on `PATH`. Run each Joern worker in its own working directory. Set `GRAPHT5_GENERATION_WORKDIR` to retain the `.sequelizer` cache between runs; otherwise the wrapper uses a temporary directory.

Use absolute input and output paths. The diff command accepts the `gen_tags.py` arguments; the VDG command accepts an intermediate JSON or JSONL dataset with `passages` as source text, and writes node-link VDGs:

```bash
bash generation/run.sh diff --input /abs/raw-pairs.json --output /abs/diff.txt --json_output /abs/paired.json --lang cpp --flat
bash generation/run.sh vdg /abs/intermediate.json /abs/with-vdgs.json --lang cpp --workers 4
bash generation/run.sh vul4c --source-dir /abs/Vul4C/source --manifest-path /abs/Vul4C/source/Vul4C.json --output-path /abs/vul4c-test.jsonl
```

The first command implements tree-sitter tokenization and `difflib` repair-instruction conversion. The VDG command calls the Joern CPG parser, Scala slice/serializer, and node-link injection. The Vul4C converter uses the same diff/tokenization routines and passage chunker. The prepared files in `../datasets/` are the inputs used for the reported results.