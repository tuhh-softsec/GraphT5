import argparse
import ast
import csv
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence
CONTROL_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "try",
    "else",
    "do",
    "synchronized",
}


@dataclass(frozen=True)
class SourceSpan:
    line_start: int
    line_end: int
    char_start: int
    char_end: int


@dataclass(frozen=True)
class TaggedSample:
    cwe_id: str
    original_address: str
    code: str
    lang: str
    source_code: str
    source_file_path: str
    source_span: SourceSpan
    source_is_full_file: bool
    predicted_cwe: str = ""
    warning_type: str = ""
    warning_msg: str = ""
    filename: str = ""

@dataclass(frozen=True)
class Span:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def overlap(self, lines: set[int]) -> int:
        return sum(1 for line in lines if self.start <= line <= self.end)


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def line_count(text: str) -> int:
    return max(1, len(normalize_newlines(text).splitlines()))


def build_source_span(source_code: str, line_start: int, line_end: int) -> SourceSpan:
    normalized = normalize_newlines(source_code)
    lines = normalized.splitlines(keepends=True)
    if not lines:
        return SourceSpan(line_start=1, line_end=1, char_start=0, char_end=0)

    start = min(max(1, line_start), len(lines))
    end = min(max(start, line_end), len(lines))
    char_start = sum(len(line) for line in lines[: start - 1])
    char_end = sum(len(line) for line in lines[:end])
    return SourceSpan(
        line_start=start,
        line_end=end,
        char_start=char_start,
        char_end=char_end,
    )


def extract_positions_start_line(row: dict[str, str]) -> int | None:
    positions = parse_python_literal(row.get("positions", ""), {})
    start_line = positions.get("start_line")
    if start_line in (None, ""):
        return None
    return int(start_line)


def infer_sast_source_span(
    row: dict[str, str], source_code: str, vulnerable_lines: Sequence[int]
) -> SourceSpan:
    snippet_lines = line_count(source_code)
    full_vul_line = extract_positions_start_line(row)
    if full_vul_line is None or not vulnerable_lines:
        return SourceSpan(
            line_start=1,
            line_end=snippet_lines,
            char_start=0,
            char_end=len(source_code),
        )

    local_vul_line = min(int(line_no) for line_no in vulnerable_lines)
    line_start = max(1, full_vul_line - local_vul_line + 1)
    line_end = line_start + snippet_lines - 1
    return SourceSpan(
        line_start=line_start,
        line_end=line_end,
        char_start=0,
        char_end=len(source_code),
    )


def normalize_cwe(cwe_id: str | None) -> str:
    value = (cwe_id or "").strip()
    if not value:
        return "CWE-000"
    if value.upper().startswith("CWE-"):
        return value.upper()
    return f"CWE-{value}"


def parse_python_literal(value: str, default):
    if not value:
        return default
    return ast.literal_eval(value)


def tokenize_and_detokenize_for_chunking(code: str, lang: str) -> str:
    from code_graph.data_utils.mod.source_tokenizer import (
        detokenize,
        tokenize_with_tree_sitter,
    )

    tokens = tokenize_with_tree_sitter(
        code,
        lang=lang,
        line_tags=True,
        preserve_markers=True,
    )
    return detokenize(tokens)


@lru_cache(maxsize=1)
def _get_chunk_tokenizer():
    from code_graph.data import GRAPHT5_TOKENS
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "Salesforce/codet5-base",
        local_files_only=True,
        use_fast=True,
    )
    tokenizer.add_tokens(GRAPHT5_TOKENS)
    return tokenizer


def _count_prefixed_tokens(tokenizer, prefix: str, text: str) -> int:
    return len(tokenizer.encode(f"{prefix} {text}".strip(), add_special_tokens=True))


def chunk_vulnerable_passages(
    vulnerable_func: str, cwe_id: str, max_tokens: int = 512
) -> list[dict]:
    prefix = f"{normalize_cwe(cwe_id)} Code Input Vulnerable Code Is:"
    tokenizer = _get_chunk_tokenizer()

    prefix_tokens = _count_prefixed_tokens(tokenizer, prefix, "")
    available = max_tokens - prefix_tokens
    if available < 1:
        raise ValueError(f"Prefix exceeds max token budget: {prefix}")

    token_ids = tokenizer.encode(vulnerable_func, add_special_tokens=False)
    if not token_ids:
        return [{"id": 0, "prefix": prefix, "text": ""}]

    passages: list[dict] = []
    start = 0
    while start < len(token_ids):
        end = min(start + available, len(token_ids))
        chunk_text = tokenizer.decode(
            token_ids[start:end],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        while (
            end > start + 1
            and _count_prefixed_tokens(tokenizer, prefix, chunk_text) > max_tokens
        ):
            end -= 1
            chunk_text = tokenizer.decode(
                token_ids[start:end],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        passages.append({"id": len(passages), "prefix": prefix, "text": chunk_text})
        start = end

    return passages


def tag_vulnerable_lines(code: str, vulnerable_lines: Sequence[int]) -> str:
    lines = normalize_newlines(code).splitlines()
    vulnerable = {int(line_no) for line_no in vulnerable_lines}
    tagged_lines = []

    for line_no, line in enumerate(lines, start=1):
        if line_no in vulnerable:
            tagged_lines.append(f"<vul-start> {line} <vul-end>")
        else:
            tagged_lines.append(line)
    return "\n".join(tagged_lines)


def extract_sast_vulnerable_lines(row: dict[str, str]) -> list[int]:
    line_range = parse_python_literal(row["vul_lines"], {})
    start = int(line_range["start_line"])
    end = int(line_range["end_line"])
    return list(range(start, end + 1))


def extract_wp7_vulnerable_lines(row: dict[str, str]) -> list[int]:
    return [int(line_no) for line_no in parse_python_literal(row["vul_nodes_list"], [])]


def choose_anchor_line(lines: Sequence[int]) -> list[int]:
    if not lines:
        return []
    ordered = sorted(set(int(line_no) for line_no in lines))
    return [ordered[len(ordered) // 2]]


def extract_wp7_focus_lines(row: dict[str, str]) -> list[int]:
    vulnerable_lines = sorted(set(extract_wp7_vulnerable_lines(row)))
    if vulnerable_lines:
        return vulnerable_lines
    fallback = [int(line_no) for line_no in parse_python_literal(row["line_numbers"], [])]
    return choose_anchor_line(fallback)


def compute_block_spans(code: str) -> list[Span]:
    lines = normalize_newlines(code).splitlines()
    stack: list[int] = []
    spans: list[Span] = []
    in_block_comment = False

    for line_no, line in enumerate(lines, start=1):
        in_string: str | None = None
        escaped = False
        i = 0
        while i < len(line):
            ch = line[i]
            nxt = line[i + 1] if i + 1 < len(line) else ""

            if in_block_comment:
                if ch == "*" and nxt == "/":
                    in_block_comment = False
                    i += 2
                    continue
                i += 1
                continue

            if in_string is not None:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == in_string:
                    in_string = None
                i += 1
                continue

            if ch == "/" and nxt == "/":
                break
            if ch == "/" and nxt == "*":
                in_block_comment = True
                i += 2
                continue
            if ch in {'"', "'"}:
                in_string = ch
                i += 1
                continue
            if ch == "{":
                stack.append(line_no)
            elif ch == "}" and stack:
                start = stack.pop()
                spans.append(Span(start=start, end=line_no))
            i += 1
    return spans


def get_signature_context(lines: list[str], span_start: int) -> tuple[int, str]:
    lower_bound = max(1, span_start - 16)
    signature_start = span_start
    signature_lines: list[str] = []
    saw_paren = False

    for candidate in range(span_start, lower_bound - 1, -1):
        line = lines[candidate - 1]
        line = line.split("{", 1)[0] if candidate == span_start else line
        stripped = line.strip()

        if not stripped:
            if saw_paren:
                break
            continue

        if (
            stripped.startswith("package ")
            or stripped.startswith("import ")
            or stripped.startswith("//")
            or stripped.startswith("*")
        ):
            if saw_paren:
                break
            continue

        if ";" in stripped or stripped.endswith("}"):
            if saw_paren:
                break
            continue

        signature_start = candidate
        signature_lines.insert(0, stripped)
        if "(" in stripped:
            saw_paren = True

    if not saw_paren:
        return span_start, ""

    while signature_start > 1 and lines[signature_start - 2].strip().startswith("@"):
        signature_start -= 1
        signature_lines.insert(0, lines[signature_start - 1].strip())

    return signature_start, " ".join(signature_lines)


def is_method_like_span(lines: list[str], span: Span) -> bool:
    span_start_line = lines[span.start - 1].split("{", 1)[0].strip()
    if any(
        span_start_line.startswith(f"{keyword} ") or span_start_line == keyword
        for keyword in CONTROL_KEYWORDS
    ):
        return False

    _, signature = get_signature_context(lines, span.start)
    if "(" not in signature or "->" in signature:
        return False

    before_paren = signature.split("(", 1)[0]
    match = re.search(r"([A-Za-z_$][A-Za-z0-9_$]*)\s*$", before_paren)
    if match is None:
        return False
    name = match.group(1)
    if name in CONTROL_KEYWORDS:
        return False
    return not re.search(
        rf"\b(class|interface|enum|record)\s+{re.escape(name)}\s*$",
        before_paren,
    )


def find_signature_start(lines: list[str], span_start: int) -> int:
    signature_start, _ = get_signature_context(lines, span_start)
    return signature_start


def select_focus_span(
    code: str, focus_lines: Sequence[int], method_only: bool = False
) -> Span | None:
    lines = normalize_newlines(code).splitlines()
    if not lines:
        return None

    valid_focus = {line_no for line_no in focus_lines if 1 <= line_no <= len(lines)}
    if not valid_focus:
        return None

    spans = compute_block_spans(code)
    if not spans:
        return None

    ranked: list[tuple[int, int, int, Span]] = []
    method_ranked: list[tuple[int, int, int, Span]] = []
    for span in spans:
        overlap = span.overlap(valid_focus)
        if overlap == 0:
            continue
        ranking = (overlap, -span.length, -span.start, span)
        ranked.append(ranking)
        if is_method_like_span(lines, span):
            method_ranked.append(ranking)

    selected = method_ranked if method_ranked else ([] if method_only else ranked)
    if not selected:
        return None
    selected.sort(reverse=True)
    return selected[0][-1]


def extract_focus_region(code: str, focus_lines: Sequence[int]) -> tuple[str, int]:
    lines = normalize_newlines(code).splitlines()
    if not lines:
        return code, 1

    span = select_focus_span(code, focus_lines)
    if span is None:
        return normalize_newlines(code), 1

    start = find_signature_start(lines, span.start)
    end = span.end
    return "\n".join(lines[start - 1 : end]), start


def extract_method_focus_region(code: str, focus_lines: Sequence[int]) -> tuple[str, int] | None:
    lines = normalize_newlines(code).splitlines()
    if not lines:
        return None

    span = select_focus_span(code, focus_lines, method_only=True)
    if span is None:
        return None

    start = find_signature_start(lines, span.start)
    end = span.end
    return "\n".join(lines[start - 1 : end]), start


def load_sast_samples(csv_path: Path) -> list[TaggedSample]:
    samples: list[TaggedSample] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source_code = normalize_newlines(row["code"])
            vulnerable_lines = extract_sast_vulnerable_lines(row)
            tagged_code = tag_vulnerable_lines(source_code, vulnerable_lines)
            repo = row.get("repo", "").strip()
            commit = row.get("commit_sha", "").strip()
            original_address = f"{repo}/commit/{commit}" if repo and commit else repo
            source_file_path = row.get("filepath", "").strip() or row.get("filename", "").strip()
            source_span = infer_sast_source_span(row, source_code, vulnerable_lines)
            samples.append(
                TaggedSample(
                    cwe_id=normalize_cwe(row.get("cwe_id")),
                    original_address=original_address,
                    code=tagged_code,
                    lang="java",
                    source_code=source_code,
                    source_file_path=source_file_path,
                    source_span=source_span,
                    source_is_full_file=False,
                    predicted_cwe=normalize_cwe(row.get("cwe_id")),
                    warning_type=(row.get("warning_type") or "").strip(),
                    warning_msg=(row.get("warning_msg") or "").strip(),
                    filename=(row.get("filename") or Path(source_file_path).name).strip(),
                )
            )
    return samples


def load_wp7_samples(csv_path: Path) -> list[TaggedSample]:
    samples: list[TaggedSample] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            status = row.get("vul_status", "").strip().lower()
            if status and status != "vul":
                continue

            vulnerable_lines = sorted(set(extract_wp7_vulnerable_lines(row)))
            if not vulnerable_lines:
                continue

            source_code = normalize_newlines(row["code_return"])
            focused_region = extract_method_focus_region(source_code, vulnerable_lines)
            if focused_region is None:
                continue

            focused_code, start_line = focused_region
            end_line = start_line + len(focused_code.splitlines()) - 1
            vulnerable_lines_local = [
                line_no - start_line + 1
                for line_no in vulnerable_lines
                if start_line <= line_no <= end_line
            ]
            if not vulnerable_lines_local:
                continue

            tagged_code = tag_vulnerable_lines(focused_code, vulnerable_lines_local)
            if "<vul-start>" not in tagged_code or "<vul-end>" not in tagged_code:
                continue
            source_span = build_source_span(source_code, start_line, end_line)
            source_file_path = row.get("file_path", "")
            file_name = (row.get("file_name") or Path(source_file_path).name).strip()
            samples.append(
                TaggedSample(
                    cwe_id=normalize_cwe(row.get("predicted_CWE")),
                    original_address=source_file_path,
                    code=tagged_code,
                    lang="java",
                    source_code=source_code,
                    source_file_path=source_file_path,
                    source_span=source_span,
                    source_is_full_file=True,
                    predicted_cwe=normalize_cwe(row.get("predicted_CWE")),
                    filename=file_name,
                )
            )
    return samples


def assemble_samples(samples: Iterable[TaggedSample]) -> list[dict]:
    dataset: list[dict] = []
    for sample_id, sample in enumerate(samples):
        vulnerable_func = tokenize_and_detokenize_for_chunking(sample.code, sample.lang)
        passage_source = vulnerable_func.replace("<line-end>", " ")
        passages = chunk_vulnerable_passages(passage_source, sample.cwe_id)
        target = f"{sample.cwe_id} Fixed Code Lines are:"
        dataset.append(
            {
                "id": sample_id,
                "cwe_id": sample.cwe_id,
                "original_address": sample.original_address,
                "vulnerable_func": vulnerable_func,
                "source_code": sample.source_code,
                "source_file_path": sample.source_file_path,
                "source_span": {
                    "line_start": sample.source_span.line_start,
                    "line_end": sample.source_span.line_end,
                    "char_start": sample.source_span.char_start,
                    "char_end": sample.source_span.char_end,
                },
                "source_is_full_file": sample.source_is_full_file,
                "predicted_cwe": sample.predicted_cwe,
                "warning_type": sample.warning_type,
                "warning_msg": sample.warning_msg,
                "filename": sample.filename,
                "passages": passages,
                "target": target,
            }
        )
    return dataset


def write_jsonl(samples: Iterable[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")


def resolve_csv(source_dir: Path, default_name: str) -> Path:
    default_csv = source_dir / default_name
    if default_csv.exists():
        return default_csv
    csv_files = list(source_dir.glob("*.csv"))
    if len(csv_files) != 1:
        raise ValueError(f"Expected exactly one CSV in {source_dir}, found {len(csv_files)}")
    return csv_files[0]


def process_source(
    *,
    source_dir: Path,
    output_path: Path,
    default_csv_name: str,
    loader,
) -> int:
    csv_path = resolve_csv(source_dir, default_csv_name)
    raw_samples = loader(csv_path)
    dataset = assemble_samples(raw_samples)
    write_jsonl(dataset, output_path)
    return len(dataset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Sec4AISEC GraphT5 test datasets from source CSV files."
    )
    parser.add_argument(
        "--sast-source",
        type=Path,
        default=Path("data/grapht5/sec4aisec/sast/source"),
    )
    parser.add_argument(
        "--wp7-source",
        type=Path,
        default=Path("data/grapht5/sec4aisec/wp7_pilot/source"),
    )
    parser.add_argument(
        "--sast-output",
        type=Path,
        default=Path("data/grapht5/sec4aisec/sast/test.jsonl"),
    )
    parser.add_argument(
        "--wp7-output",
        type=Path,
        default=Path("data/grapht5/sec4aisec/wp7_pilot/test.jsonl"),
    )
    args = parser.parse_args()

    sast_count = process_source(
        source_dir=args.sast_source,
        output_path=args.sast_output,
        default_csv_name="jsqlparser_dataset_cwe.csv",
        loader=load_sast_samples,
    )
    wp7_count = process_source(
        source_dir=args.wp7_source,
        output_path=args.wp7_output,
        default_csv_name="pilot.csv",
        loader=load_wp7_samples,
    )

    print(f"Wrote {sast_count} samples to {args.sast_output}")
    print(f"Wrote {wp7_count} samples to {args.wp7_output}")


if __name__ == "__main__":
    main()
