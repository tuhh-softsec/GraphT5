import re
from typing import List

import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

CPP_LANGUAGE = Language(tscpp.language())
C_LANGUAGE = Language(tsc.language())
JAVA_LANGUAGE = Language(tsjava.language())
PARSER = Parser()


def tokenize_with_tree_sitter(
    code_str: str,
    lang: str = "java",
    line_tags=True,
    preserve_markers=False,
    preserve_unicode=False,
) -> List[str]:
    """
    Tokenize code using tree-sitter, encoding <vul-start> and <vul-end> tags.
    """

    if lang == "java":
        parser = Parser(JAVA_LANGUAGE)
    elif lang == "cpp":
        parser = Parser(CPP_LANGUAGE)
    elif lang == "c":
        parser = Parser(C_LANGUAGE)
    else:
        raise ValueError(f"Unsupported language: {lang}")

    code_str = code_str.replace("<vul-start>", "/*VULSTART*/").replace(
        "<vul-end>", "/*VULEND*/"
    )
    code_str = code_str.replace("<null>", "/*NULLTOKEN*/").replace("<line-end>", "\n")

    if not preserve_unicode:
        code_str = "".join(c if ord(c) < 128 else "_" for c in code_str)

    code_str = normalize_backslash_whitespace(code_str)

    code_bytes = code_str.encode("utf-8")
    tree = parser.parse(code_bytes)

    tokens = []

    def is_line_break():
        if tokens:
            return tokens[-1] == "/*LINEEND*/"
        return False

    def traverse(node, source_bytes, last_row=0, leaf_sequence=None):
        if not node.children:
            token = _process_node(node, source_bytes)

            if (
                node.type == "line_comment"
                or node.type == "block_comment"
                or node.type == "comment"
            ):
                if line_tags and token == "/*LINECONTINUATION*/":
                    tokens.append("/*LINECONTINUATION*/")
                elif preserve_markers:
                    if token in ["/*VULSTART*/", "<vul-start>"]:
                        tokens.append("<vul-start>")
                    elif token in ["/*VULEND*/", "<vul-end>"]:
                        tokens.append("<vul-end>")
            elif node.type == "preproc_arg":
                # parse the macro body as C++ and extract tokens
                try:
                    temp_parser = Parser(CPP_LANGUAGE)
                    # we need to preserve line continuations in macro bodies
                    token = token.replace("\\\n", "/*LINECONTINUATION*/")

                    token_bytes = token.encode("utf-8")
                    token_tree = temp_parser.parse(token_bytes)

                    traverse(token_tree.root_node, token_bytes)
                except Exception:
                    # fallback: treat as single token
                    tokens.append(token)
            elif token:
                current_row = node.start_point.row
                if line_tags and last_row < current_row and not is_line_break():
                    tokens.append("/*LINEEND*/")

                leaf_sequence = leaf_sequence + token if leaf_sequence else token

                if not leaf_sequence.startswith('"') or node.next_sibling is None:
                    tokens.append(leaf_sequence)

                    if line_tags and node.type == "#endif":
                        tokens.append("/*LINEEND*/")

                    leaf_sequence = None

                # tokens.append(token)
                last_row = node.end_point.row
        else:
            remaining_nodes = None
            if node.type == "preproc_function_def":
                preproc_function_def_tokens, remaining_nodes = (
                    _process_preproc_function_def(node, source_bytes)
                )

                if line_tags and not is_line_break():
                    tokens.append("/*LINEEND*/")

                tokens.extend(preproc_function_def_tokens)

            remaining_children = remaining_nodes or node.children
            for child in remaining_children:
                last_row, leaf_sequence = traverse(
                    child, source_bytes, last_row, leaf_sequence
                )

        return last_row, leaf_sequence

    traverse(tree.root_node, code_bytes)

    return tokens


def _process_node(node, source_bytes):
    token = source_bytes[node.start_byte : node.end_byte]
    token = token.decode("utf-8")
    token = _replace_tokens(token)

    return token.strip()


def _replace_tokens(token):
    token = token.replace("/*VULSTART*/", "<vul-start>").replace(
        "/*VULEND*/", "<vul-end>"
    )
    token = token.replace("/*NULLTOKEN*/", "<null>")
    return token


def _process_preproc_function_def(node, source_bytes):
    assert len(node.children) >= 3

    define_field = node.children[0]
    identifier_field = node.children[1]
    preproc_parameters = node.children[2]

    assert define_field.type == "#define"
    assert identifier_field.type == "identifier"
    assert preproc_parameters.type == "preproc_params"

    preproc_def_tokens = []

    define_token = _process_node(define_field, source_bytes)
    preproc_function_def_node = source_bytes[
        identifier_field.start_byte : preproc_parameters.end_byte
    ]

    preproc_function_def_token = preproc_function_def_node.decode("utf-8")
    preproc_function_def_token = _replace_tokens(preproc_function_def_token)

    preproc_def_tokens.append(define_token)
    preproc_def_tokens.append(preproc_function_def_token)

    return preproc_def_tokens, node.children[3:]


def normalize_backslash_whitespace(text):
    """
    Normalize whitespace around single backslashes in already decoded text
    """
    # Clean multiple spaces/tabs before and after backslash
    # \s* matches any whitespace (spaces, tabs, etc.)
    text = re.sub(r"\s*\\\s*\n", " \\\n", text)

    return text


def tokenize_source(
    code_str: str, lang: str = "java", line_tags=False, preserve_markers=False
) -> List[str]:
    return tokenize_with_tree_sitter(
        code_str, lang, line_tags=line_tags, preserve_markers=preserve_markers
    )


def tokenize_java(code_str: str) -> List[str]:
    return tokenize_source(code_str, "java")


def tokenize_cpp(code_str: str) -> List[str]:
    return tokenize_source(code_str, "cpp")


def _join_character_literals(tokens: List[str]) -> List[str]:
    joined = []
    index = 0
    prefixes = {"'", "L'", "u'", "U'", "u8'"}

    while index < len(tokens):
        token = tokens[index]
        if token not in prefixes:
            joined.append(token)
            index += 1
            continue

        end = index + 1
        while end < len(tokens) and tokens[end] != "'":
            end += 1
        if end == len(tokens):
            joined.append(token)
            index += 1
            continue

        joined.append("".join(tokens[index : end + 1]))
        index = end + 1

    return joined


def detokenize(tokens: List[str]) -> str:
    return " ".join(_join_character_literals(tokens)).replace(
        "/*LINEEND*/", "<line-end>"
    )
