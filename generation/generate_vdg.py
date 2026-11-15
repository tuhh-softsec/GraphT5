import argparse
from pathlib import Path

from code_graph.data import (
    SegmentType,
    extract_segments,
    extract_vulnerable_code_from_java,
    inject_segments,
    load_json_file,
    save_dataset_to_file,
)
from code_graph.vdg import generate_vdg_for_segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lang", choices=("c", "cpp", "java"), default="cpp")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    dataset = load_json_file(args.dataset)
    segments = (
        extract_vulnerable_code_from_java(dataset)
        if args.lang == "java"
        else extract_segments(dataset, SegmentType.VULNERABLE_CODE_INPUT)
    )
    graphs = generate_vdg_for_segments(
        segments, num_proc=args.workers, node_link=True, lang=args.lang
    )
    prepared = inject_segments(
        dataset, graphs, SegmentType.VULNERABLE_VDG_INPUT, node_link=True
    )
    save_dataset_to_file(args.output, prepared)


if __name__ == "__main__":
    main()
