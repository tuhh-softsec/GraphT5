# @Time: 2025.4.29 14:51
# @Author: Noah Frontzkowski


import json
import logging
import os
import re
import subprocess
import uuid
import warnings
from pathlib import Path
from typing import List, Optional

import networkx as nx

from ..util import OUTPUT_DIR, define_all_preprocessors_checks, move_vul_tag
from .cpg import CPGServer
from .vdg import VDGGenerator

warnings.simplefilter(action="ignore", category=FutureWarning)


logger = logging.getLogger(__name__)

# constants
helpers_dir = os.path.dirname(__file__)

DECLARATION_MACROS = (
    (
        re.compile(
            r"(?m)^([ \t]*)DECLAREcpFunc[ \t]*\([ \t]*(\w+)[ \t]*\)(?=[ \t]*(?:\n[ \t]*)?\{)"
        ),
        r"\1static int \2(TIFF* in, TIFF* out, uint32 imagelength, uint32 imagewidth, tsample_t spp)",
    ),
    (
        re.compile(
            r"(?m)^([ \t]*)DECLAREreadFunc[ \t]*\([ \t]*(\w+)[ \t]*\)(?=[ \t]*(?:\n[ \t]*)?\{)"
        ),
        r"\1static int \2(TIFF* in, uint8* buf, uint32 imagelength, uint32 imagewidth, tsample_t spp)",
    ),
    (
        re.compile(r"(?m)^([ \t]*)IW_IMPL[ \t]*\([ \t]*([^)]+?)[ \t]*\)"),
        r"\1\2",
    ),
    (
        re.compile(r"(?m)^([ \t]*)METHODDEF[ \t]*\([ \t]*([^)]+?)[ \t]*\)"),
        r"\1static \2",
    ),
    (re.compile(r"(?m)^([ \t]*)GLOBAL[ \t]*\([ \t]*([^)]+?)[ \t]*\)[ \t]*$"), r"\1\2"),
    (
        re.compile(r"(?m)^([ \t]*)LOCAL[ \t]*\([ \t]*([^)]+?)[ \t]*\)[ \t]*$"),
        r"\1static \2",
    ),
)


def normalize_c_family_declarations(source: str) -> str:
    for pattern, replacement in DECLARATION_MACROS:
        source = pattern.sub(replacement, source)
    return source


def _build_cpg_generation_params(lang: str):
    if lang in ["c", "cpp"]:
        return {
            "include-comments": "",
            "with-include-auto-discovery": "",
            "no-image-locations": "",
            "define": [
                "UNUSED_ATTR",
                "UNUSED",
                "ATTRIBUTE_UNUSED",
                "_U_",
                "_GL_UNUSED",
                "__rte_unused",
                "__user",
                "__init",
                "__net_init",
                "FLAC__",
                "noinline",
                "__forceinline",
                "__always_inline",
                "__rte_always_inline",
                "JERRY_ATTR_NOINLINE",
                "checkreturn",
                "STACK_OF(type) struct stack_st_##type",
                "URI_FUNC(func) func",
                "URI_TYPE(type) type",
                "STATIC",
                "CLASS",
                "FAR",
                "R_OWN",
                "ZLIB_INTERNAL",
                "BROTLI_INLINE",
                "_Xconst",
                "TSRMLS_DC",
                "coroutine_fn",
            ],
        }
    return None


def generate_cpgs(
    code_strs: List[str], ids: Optional[List[str]] = None, lang: Optional[str] = "cpp"
):
    """Generate CPG files using Joern."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with CPGServer(lang=lang, verbose=False) as server:
        for index, code_str in enumerate(code_strs):
            segment_id = ids[index] if ids and index < len(ids) else str(uuid.uuid4())

            if lang in ["c", "cpp"]:
                code_str = normalize_c_family_declarations(code_str)

            temp_file_path = os.path.join(OUTPUT_DIR, f"{segment_id}.{lang}")
            cpg_file_path = Path(temp_file_path.replace(f".{lang}", ".cpg.bin.zip"))

            if not ids or not (
                cpg_file_path.exists() and os.path.exists(temp_file_path)
            ):
                with open(temp_file_path, "w", encoding="utf-8") as temp_file:
                    temp_file.write(code_str)

                # format the source without aligning commands
                subprocess.run(
                    f'clang-format -style="{{ColumnLimit: 99999999, AlignTrailingComments: false, IndentPPDirectives: '
                    f"None, ReflowComments: false, AlignEscapedNewlines: DontAlign,"
                    f"BreakBeforeBraces: Attach, BreakAfterAttributes: Never,"  # BreakBinaryOperations: Never, "
                    f"BreakStringLiterals: false,"
                    f"AllowAllArgumentsOnNextLine: true, AllowAllParametersOfDeclarationOnNextLine: true,"
                    f"BinPackArguments: true, BinPackParameters: true,"
                    f"AllowShortBlocksOnASingleLine: Never,"  # AllowShortCaseExpressionOnASingleLine: false, "
                    f"AllowShortCaseLabelsOnASingleLine: false, AllowShortCompoundRequirementOnASingleLine: false, "
                    f"AllowShortEnumsOnASingleLine: false, AllowShortFunctionsOnASingleLine: None, "
                    f"AllowShortIfStatementsOnASingleLine: Never, AllowShortLambdasOnASingleLine: None, "
                    f'AllowShortLoopsOnASingleLine: false}}" -i "{temp_file_path}"',
                    shell=True,
                    check=True,
                )

                # move trailing <vul-start> tags to the next line
                define_all_preprocessors_checks(temp_file_path)
                move_vul_tag(temp_file_path)

                # generate CPG
                server.generate_cpg(
                    temp_file_path,
                    str(cpg_file_path),
                    params=_build_cpg_generation_params(lang),
                )

                assert os.path.exists(cpg_file_path), (
                    f"{cpg_file_path}: FAIL. CPG file does not exist."
                )

            yield cpg_file_path, temp_file_path


def run_joern(
    source_files: list,
    cpg_files: list,
    stats_base_paths: list = None,
    lang: Optional[str] = "cpp",
    verbose=False,
):
    """Use Joern to generate VDGs of vulnerable CPGs."""
    assert not stats_base_paths or len(cpg_files) == len(stats_base_paths), (
        "cpg_files and stats_base_paths must have the same length!"
    )

    with VDGGenerator(lang=lang, verbose=verbose) as vdg_bridge:
        for node_link_file in vdg_bridge.generate_vdgs(
            source_files, cpg_files, stats_base_paths
        ):
            if node_link_file:
                logger.debug(
                    f"Successfully generated and validated VDG: {node_link_file}"
                )

                yield node_link_file
            else:
                logger.error(f"Failed to generate or validate VDG for {node_link_file}")
                yield None


def parse_joern_vdg(filepath):
    with open(filepath, "r") as f:
        node_link_data = json.load(f)
        vdg = nx.node_link_graph(node_link_data)

    return vdg


def parse_joern_node_link(filepath):
    if filepath:
        with open(filepath, "r") as f:
            node_link_data = json.load(f)
    else:
        node_link_data = {
            "directed": True,
            "multigraph": True,
            "graph": {},
            "nodes": [],
            "links": [],
        }

    return node_link_data


def _run_joern_c2cpg(filepath: str):
    path = Path(filepath)
    workspace_dir = path.resolve().parent
    workspace_fil = workspace_dir / (path.stem + ".cpg.bin.zip")

    command = (
        f'c2cpg.sh "{filepath}" --output "{workspace_fil}" --include-comments --with-include-auto-discovery '
        f"--no-image-locations"
    )

    result = subprocess.run(
        command,
        shell=True,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Command '{command}' failed with exit code {result.returncode}")
        logger.error(f"stderr: {result.stderr}")

    return workspace_fil
