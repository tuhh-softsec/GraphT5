import argparse
from pathlib import Path

from loguru import logger

from grapht5.datasets.packing import pack_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    stats = pack_jsonl(args.source, args.destination)
    reduction = 1.0 - stats.output_bytes / stats.input_bytes
    logger.info(
        "Packed {} samples: {:.2f} GB to {:.2f} GB ({:.1%} smaller)",
        stats.samples,
        stats.input_bytes / 1e9,
        stats.output_bytes / 1e9,
        reduction,
    )


if __name__ == "__main__":
    main()
