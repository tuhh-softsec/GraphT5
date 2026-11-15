from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import msgspec
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
from loguru import logger
from tree_sitter import Language, Node, Parser

from code_graph.data_utils.mod.gen_tags import get_token_pair_diff
from code_graph.data_utils.mod.source_tokenizer import tokenize_source
from code_graph.data import GRAPHT5_TOKENS
from code_graph.data_utils.chunk import DatasetChunker


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.append(str(REPO_ROOT))

from scripts.build_sec4aisec_dataset import chunk_vulnerable_passages

METADATA_RE = re.compile(r"(?P<cve>CVE-\d{4}-\d+)_(?P<cwe>[^_]+)_")
CPP_SUFFIXES = {".cc", ".cpp", ".cxx", ".c++", ".hpp", ".hh", ".hxx"}
FUNCTION_NODE_TYPES = {"function_definition", "method_definition"}
NAME_NODE_TYPES = {
    "destructor_name",
    "field_identifier",
    "identifier",
    "operator_name",
    "qualified_identifier",
}
C_LANGUAGE = Language(tsc.language())
CPP_LANGUAGE = Language(tscpp.language())
VDG_SOURCE_KEY = "_vdg_source"


@dataclass(frozen=True)
class Vul4CPair:
    old_path: Path
    new_path: Path
    relative_old_path: Path
    cve_id: str
    cwe_id: str
    lang: str


@dataclass(frozen=True)
class SourceFunction:
    name: str
    code: str
    start_line: int
    end_line: int
    ordinal: int


@dataclass(frozen=True)
class RepairGroup:
    repair_group_id: str
    changed_files: tuple[str, ...]

    @property
    def multi_file_patch(self) -> bool:
        return len(self.changed_files) > 1


@dataclass(frozen=True)
class FunctionPairingResult:
    pairs: list[tuple[SourceFunction, SourceFunction]]
    old_unmatched: list[SourceFunction]
    new_unmatched: list[SourceFunction]
    old_candidates: list[SourceFunction]
    new_candidates: list[SourceFunction]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Vul4C OLD/NEW source pairs to GraphT5 test.jsonl."
    )
    parser.add_argument("--source-dir", type=Path, default=SCRIPT_DIR / "source")
    parser.add_argument(
        "--manifest-path", type=Path, default=SCRIPT_DIR / "source" / "Vul4C.json"
    )
    parser.add_argument("--output-path", type=Path, default=SCRIPT_DIR / "test.jsonl")
    parser.add_argument("--max-passage-tokens", type=int, default=512)
    parser.add_argument("--max-passages", type=int, default=15)
    parser.add_argument("--trim-padding", type=int, default=512)
    parser.add_argument(
        "--all-file-pairs",
        action="store_true",
        help="Convert every changed file pair instead of the 144 manifest entries.",
    )
    parser.add_argument(
        "--skip-vdg-generation",
        action="store_true",
        help="Write empty VDG placeholders instead of generating node-link VDGs.",
    )
    parser.add_argument("--vdg-workers", type=int, default=8)
    parser.add_argument("--vdg-work-dir", type=Path)
    parser.add_argument("--vdg-stats-dir", type=Path)
    parser.add_argument("--vdg-lang", choices=["cpp", "java"], default="cpp")
    parser.add_argument("--verbose-vdg", action="store_true")
    return parser.parse_args()


def infer_lang(path: Path) -> str:
    suffixes = {suffix.lower() for suffix in path.suffixes}
    if suffixes & CPP_SUFFIXES:
        return "cpp"
    return "c"


def parse_metadata(path: Path) -> tuple[str, str]:
    match = METADATA_RE.search(path.name)
    assert match is not None, f"Cannot parse CVE/CWE metadata from {path}"
    return match.group("cve"), match.group("cwe")


def is_old_file(path: Path) -> bool:
    return "_OLD" in path.name or ".OLD" in path.name


def is_new_file(path: Path) -> bool:
    return "_NEW" in path.name or ".NEW" in path.name


def new_path_for_old(old_path: Path) -> Path:
    if "_OLD" in old_path.name:
        return old_path.with_name(old_path.name.replace("_OLD", "_NEW", 1))
    if ".OLD_" in old_path.name:
        return old_path.with_name(old_path.name.replace(".OLD_", "_NEW.", 1))
    return old_path.with_name(old_path.name.replace(".OLD", ".NEW", 1))


def changed_file_key(path: Path) -> str:
    name = path.name
    if "_OLD" in name:
        normalized_name = name.split("_OLD", 1)[0]
    else:
        normalized_name = name.split(".OLD", 1)[0]
    return (path.parent / normalized_name).as_posix()


def build_pair(source_dir: Path, old_path: Path) -> Vul4CPair:
    new_path = new_path_for_old(old_path)
    assert new_path.exists(), f"Expected NEW file for {old_path}"
    cve_id, cwe_id = parse_metadata(old_path)
    return Vul4CPair(
        old_path=old_path,
        new_path=new_path,
        relative_old_path=old_path.relative_to(source_dir),
        cve_id=cve_id,
        cwe_id=cwe_id,
        lang=infer_lang(old_path),
    )


def find_all_file_pairs(source_dir: Path) -> list[Vul4CPair]:
    old_paths = sorted(
        (
            path
            for path in source_dir.rglob("*")
            if path.is_file() and is_old_file(path)
        ),
        key=lambda p: p.relative_to(source_dir).as_posix(),
    )
    new_paths = {
        path.relative_to(source_dir).as_posix()
        for path in source_dir.rglob("*")
        if path.is_file() and is_new_file(path)
    }
    pairs = []
    matched_new_keys = set()

    for old_path in old_paths:
        pair = build_pair(source_dir, old_path)
        pairs.append(pair)
        matched_new_keys.add(pair.new_path.relative_to(source_dir).as_posix())

    extra_new_files = sorted(new_paths - matched_new_keys)
    if extra_new_files:
        logger.warning("Ignoring {} unmatched NEW files.", len(extra_new_files))

    return pairs


def find_manifest_pairs(source_dir: Path, manifest_path: Path) -> list[Vul4CPair]:
    manifest = msgspec.json.decode(manifest_path.read_bytes())
    pairs = []

    for cve_id, item in manifest.items():
        cve_dir = source_dir / item["project"] / cve_id
        old_paths = sorted(
            path for path in cve_dir.iterdir() if path.is_file() and is_old_file(path)
        )
        assert old_paths, (
            f"Expected at least one OLD file for manifest vulnerability {cve_id}"
        )
        pairs.extend(build_pair(source_dir, path) for path in old_paths)

    return pairs


def empty_vdg() -> dict[str, Any]:
    return {"directed": True, "links": [], "nodes": [], "graph": {}, "multigraph": True}


def is_empty_vdg(vdg: dict[str, Any]) -> bool:
    return not vdg["nodes"]


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def format_passages(source: str, cwe_id: str, max_tokens: int) -> list[dict[str, Any]]:
    return format_passages_with_limit(
        source,
        cwe_id,
        max_tokens=max_tokens,
        max_passages=15,
        trim_padding=512,
    )


@lru_cache(maxsize=1)
def get_context_chunker() -> DatasetChunker:
    return DatasetChunker(tokens=GRAPHT5_TOKENS)


def format_passages_with_limit(
    source: str,
    cwe_id: str,
    max_tokens: int,
    max_passages: int,
    trim_padding: int,
) -> list[dict[str, Any]]:
    normalized_source = source.replace("<line-end>", " ")
    passages = chunk_vulnerable_passages(
        normalized_source,
        cwe_id,
        max_tokens=max_tokens,
    )
    if not max_passages or len(passages) <= max_passages:
        return passages

    chunker = get_context_chunker()
    prefix_tokens = len(
        chunker.tokenizer.encode(passages[0]["prefix"], add_special_tokens=True)
    )
    content_budget = max_passages * (max_tokens - prefix_tokens)
    trimmed_source = chunker.trim_function_to_vulnerable_chunks(
        normalized_source,
        token_limit=content_budget,
        padding=trim_padding,
    )
    passages = chunk_vulnerable_passages(
        trimmed_source,
        cwe_id,
        max_tokens=max_tokens,
    )
    if len(passages) > max_passages:
        raise RuntimeError("Vulnerability-centered source exceeds the passage budget.")
    return passages


def get_parser(lang: str) -> Parser:
    if lang == "cpp":
        return Parser(CPP_LANGUAGE)
    return Parser(C_LANGUAGE)


def node_text(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode(
        "utf-8", errors="replace"
    )


def function_name_from_declarator(node: Node | None, source_bytes: bytes) -> str | None:
    if node is None:
        return None

    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return node_text(name_node, source_bytes)

    nested_declarator = node.child_by_field_name("declarator")
    if nested_declarator is not None:
        name = function_name_from_declarator(nested_declarator, source_bytes)
        if name:
            return name

    if node.type in NAME_NODE_TYPES:
        return node_text(node, source_bytes)

    for child in node.children:
        name = function_name_from_declarator(child, source_bytes)
        if name:
            return name
    return None


def extract_functions(source_code: str, lang: str) -> list[SourceFunction]:
    source_bytes = source_code.encode("utf-8")
    tree = get_parser(lang).parse(source_bytes)
    functions = []

    def visit(node: Node) -> None:
        if node.type in FUNCTION_NODE_TYPES:
            name = function_name_from_declarator(
                node.child_by_field_name("declarator"), source_bytes
            )
            if name is not None:
                functions.append(
                    SourceFunction(
                        name=name,
                        code=node_text(node, source_bytes),
                        start_line=node.start_point.row + 1,
                        end_line=node.end_point.row + 1,
                        ordinal=len(functions),
                    )
                )
        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return functions


def changed_line_sets(old_code: str, new_code: str) -> tuple[set[int], set[int]]:
    old_lines = old_code.splitlines()
    new_lines = new_code.splitlines()
    old_changed = set()
    new_changed = set()

    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, old_lines, new_lines, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            continue

        old_changed.update(range(i1 + 1, i2 + 1))
        new_changed.update(range(j1 + 1, j2 + 1))

        if tag == "insert" and old_lines:
            old_changed.add(min(max(i1, 1), len(old_lines)))
        if tag == "delete" and new_lines:
            new_changed.add(min(max(j1, 1), len(new_lines)))

    return old_changed, new_changed


def functions_intersecting(
    functions: list[SourceFunction], lines: set[int]
) -> list[SourceFunction]:
    return [
        function
        for function in functions
        if any(function.start_line <= line <= function.end_line for line in lines)
    ]


def pair_by_name(
    old_functions: list[SourceFunction],
    new_functions: list[SourceFunction],
) -> tuple[
    list[tuple[SourceFunction, SourceFunction]],
    list[SourceFunction],
    list[SourceFunction],
]:
    old_by_name = group_functions_by_name(old_functions)
    new_by_name = group_functions_by_name(new_functions)
    pairs = []

    for name in sorted(old_by_name.keys() & new_by_name.keys()):
        old_group = old_by_name[name]
        new_group = new_by_name[name]
        assert len(old_group) == len(new_group), (
            f"Ambiguous function match for {name}: "
            f"{len(old_group)} old functions, {len(new_group)} new functions"
        )
        pairs.extend(zip(old_group, new_group))

    unmatched_old = [
        function for function in old_functions if function.name not in new_by_name
    ]
    unmatched_new = [
        function for function in new_functions if function.name not in old_by_name
    ]
    return pairs, unmatched_old, unmatched_new


def group_functions_by_name(
    functions: list[SourceFunction],
) -> dict[str, list[SourceFunction]]:
    grouped: dict[str, list[SourceFunction]] = {}
    for function in functions:
        grouped.setdefault(function.name, []).append(function)
    return grouped


def pair_changed_functions_for_lang(
    old_code: str,
    new_code: str,
    old_changed_lines: set[int],
    new_changed_lines: set[int],
    parser_lang: str,
) -> FunctionPairingResult:
    old_functions = extract_functions(old_code, parser_lang)
    new_functions = extract_functions(new_code, parser_lang)
    old_changed = functions_intersecting(old_functions, old_changed_lines)
    new_changed = functions_intersecting(new_functions, new_changed_lines)

    if not old_changed and not new_changed:
        return FunctionPairingResult([], [], [], [], [])

    matched, old_unmatched, new_unmatched = pair_by_name(old_changed, new_changed)
    return FunctionPairingResult(
        matched,
        old_unmatched,
        new_unmatched,
        old_changed,
        new_changed,
    )


def changed_function_pairs(
    pair: Vul4CPair,
) -> list[tuple[SourceFunction, SourceFunction]]:
    old_code = pair.old_path.read_text(encoding="utf-8", errors="replace")
    new_code = pair.new_path.read_text(encoding="utf-8", errors="replace")
    old_changed_lines, new_changed_lines = changed_line_sets(old_code, new_code)

    if not old_changed_lines and not new_changed_lines:
        return []

    result = pair_changed_functions_for_lang(
        old_code, new_code, old_changed_lines, new_changed_lines, pair.lang
    )
    if pair.lang == "c" and not result.pairs:
        fallback = pair_changed_functions_for_lang(
            old_code, new_code, old_changed_lines, new_changed_lines, "cpp"
        )
        if fallback.pairs:
            logger.warning(
                "Using C++ parser fallback for changed function extraction in {}.",
                pair.relative_old_path,
            )
            result = fallback

    if not result.old_candidates and not result.new_candidates:
        logger.warning(
            "Skipping {} because its changed lines are outside function definitions.",
            pair.relative_old_path,
        )
        return []

    if result.old_unmatched or result.new_unmatched:
        logger.warning(
            "Skipping unmatched changed functions in {}: old={}, new={}.",
            pair.relative_old_path,
            [function.name for function in result.old_unmatched],
            [function.name for function in result.new_unmatched],
        )
    return result.pairs


def build_samples(
    pair: Vul4CPair,
    repair_group: RepairGroup,
    first_sample_id: int,
    max_passage_tokens: int,
    max_passages: int,
    trim_padding: int,
) -> list[dict[str, Any]]:
    samples = []

    for old_function, new_function in changed_function_pairs(pair):
        vulnerable_tokens = tokenize_source(
            old_function.code, lang=pair.lang, line_tags=True
        )
        fixed_tokens = tokenize_source(
            new_function.code, lang=pair.lang, line_tags=True
        )
        source, target = get_token_pair_diff(vulnerable_tokens, fixed_tokens)

        if not source or not target:
            logger.warning(
                "Skipping {}:{} because the function-level diff is empty.",
                pair.relative_old_path,
                old_function.name,
            )
            continue

        sample_id = first_sample_id + len(samples)
        function_repair_id = (
            f"{repair_group.repair_group_id}"
            f"::{pair.relative_old_path.as_posix()}"
            f"::{old_function.name}"
            f"::{old_function.ordinal}"
        )
        samples.append(
            {
                "id": sample_id,
                "cwe_id": pair.cwe_id,
                "cve_id": pair.cve_id,
                "repair_group_id": repair_group.repair_group_id,
                "function_repair_id": function_repair_id,
                "function_name": old_function.name,
                "source_file_path": pair.relative_old_path.as_posix(),
                "multi_file_patch": repair_group.multi_file_patch,
                "num_changed_files": len(repair_group.changed_files),
                "changed_files": list(repair_group.changed_files),
                "original_address": function_repair_id,
                "passages": format_passages_with_limit(
                    source,
                    pair.cwe_id,
                    max_tokens=max_passage_tokens,
                    max_passages=max_passages,
                    trim_padding=trim_padding,
                ),
                "target": f"{pair.cwe_id} Fixed Code Lines are: {target}",
                "vdg": empty_vdg(),
                VDG_SOURCE_KEY: source,
            }
        )

    return samples


def attach_vdgs(
    samples: list[dict[str, Any]],
    work_dir: Path,
    workers: int,
    stats_dir: Path | None,
    lang: str,
    verbose: bool,
) -> None:
    from code_graph.vdg import generate_vdg_for_segments

    segments = [
        (sample["id"], sample["cwe_id"], sample[VDG_SOURCE_KEY]) for sample in samples
    ]

    with working_directory(work_dir):
        generated = generate_vdg_for_segments(
            segments,
            num_proc=workers,
            node_link=True,
            stats_dir=str(stats_dir) if stats_dir else None,
            lang=lang,
            verbose=verbose,
        )

    vdgs_by_id = {segment[0]: vdg or empty_vdg() for segment, vdg in generated}

    for sample in samples:
        sample["vdg"] = vdgs_by_id[sample["id"]]

    nonempty = sum(not is_empty_vdg(sample["vdg"]) for sample in samples)
    logger.info(
        "Attached non-empty VDGs to {}/{} Vul4C samples.", nonempty, len(samples)
    )
    if samples and nonempty == 0:
        raise RuntimeError("VDG generation produced only empty graphs.")


def strip_internal_fields(samples: list[dict[str, Any]]) -> None:
    for sample in samples:
        sample.pop(VDG_SOURCE_KEY, None)


def repair_group_id(pair: Vul4CPair) -> str:
    project, _, _ = pair.relative_old_path.parts[:3]
    return f"vul4c://{project}/{pair.cve_id}"


def build_repair_groups(pairs: list[Vul4CPair]) -> dict[str, RepairGroup]:
    files_by_group: dict[str, set[str]] = {}
    for pair in pairs:
        group_id = repair_group_id(pair)
        files_by_group.setdefault(group_id, set()).add(
            changed_file_key(pair.relative_old_path)
        )

    return {
        group_id: RepairGroup(group_id, tuple(sorted(changed_files)))
        for group_id, changed_files in files_by_group.items()
    }


def convert_dataset(
    source_dir: Path,
    output_path: Path,
    manifest_path: Path | None = None,
    max_passage_tokens: int = 512,
    max_passages: int = 15,
    trim_padding: int = 512,
    all_file_pairs: bool = False,
    generate_vdgs: bool = True,
    vdg_workers: int = 8,
    vdg_work_dir: Path | None = None,
    vdg_stats_dir: Path | None = None,
    vdg_lang: str = "cpp",
    verbose_vdg: bool = False,
) -> list[dict[str, Any]]:
    if all_file_pairs:
        pairs = find_all_file_pairs(source_dir)
    else:
        assert manifest_path is not None, (
            "manifest_path is required unless all_file_pairs is set"
        )
        pairs = find_manifest_pairs(source_dir, manifest_path)

    repair_groups = build_repair_groups(pairs)
    samples = []

    for pair in pairs:
        samples.extend(
            build_samples(
                pair,
                repair_groups[repair_group_id(pair)],
                len(samples),
                max_passage_tokens,
                max_passages,
                trim_padding,
            )
        )

    if generate_vdgs:
        attach_vdgs(
            samples,
            work_dir=vdg_work_dir or output_path.parent,
            workers=vdg_workers,
            stats_dir=vdg_stats_dir,
            lang=vdg_lang,
            verbose=verbose_vdg,
        )

    strip_internal_fields(samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        for sample in samples:
            handle.write(msgspec.json.encode(sample))
            handle.write(b"\n")

    logger.info("Converted {} Vul4C samples to {}.", len(samples), output_path)
    return samples


def main() -> None:
    args = parse_args()
    convert_dataset(
        args.source_dir,
        args.output_path,
        manifest_path=args.manifest_path,
        max_passage_tokens=args.max_passage_tokens,
        max_passages=args.max_passages,
        trim_padding=args.trim_padding,
        all_file_pairs=args.all_file_pairs,
        generate_vdgs=not args.skip_vdg_generation,
        vdg_workers=args.vdg_workers,
        vdg_work_dir=args.vdg_work_dir,
        vdg_stats_dir=args.vdg_stats_dir,
        vdg_lang=args.vdg_lang,
        verbose_vdg=args.verbose_vdg,
    )


if __name__ == "__main__":
    main()
