# Prepared datasets

Prepared GraphT5 inputs contain text passages, VDGs, and repair-instruction targets.
The tokenization, diff conversion, and VDG construction source is in `../generation/`.

| Dataset | Training | Validation | Test |
|---|---:|---:|---:|
| pretraining | 458,918 | 470 | — |
| bigvul_cvefixes | 3,584 | 292 | 1,496 |
| icvul | 3,892 | 487 | 487 |
| primevul | 3,712 | 470 | 428 |
| vul4j | 5,295 | 589 | 216 |
| vul4c | 3,678 | 470 | 272 |

The fine-tuning datasets are stored as gzip-compressed packed files (`.cjsonl.gz`). The package also accepts uncompressed `.cjsonl` files and `.jsonl` files in the object-based sample schema; all three formats load the same samples. Pass the `.cjsonl.gz` paths directly to the training and evaluation scripts. The pretraining dataset is not included; see `pretraining/README.md`.