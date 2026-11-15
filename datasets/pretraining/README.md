# Pretraining dataset

GraphT5 is pretrained on the C/C++ bug-fix corpus released by Chen et al. with VRepair:

> Zimin Chen, Steve Kommrusch, and Martin Monperrus. 2023. Neural Transfer Learning for Repairing Security Vulnerabilities in C Code. *IEEE Transactions on Software Engineering* 49, 1, 147–165. https://doi.org/10.1109/TSE.2022.3147265

The prepared GraphT5 inputs are not stored in this repository because of their size. Place them in this directory:

| File | Samples | Size |
|---|---:|---:|
| `train.cjsonl` | 458,918 | 3.5 GB |
| `valid.cjsonl` | 470 | 12 MB |

Both files use the packed `.cjsonl` schema described in `../README.md`. To build them from the original corpus, use the tokenization, diff conversion, and VDG construction source in `../../generation/`.
