import logging
import multiprocessing
import os
import re
import uuid

logger = logging.getLogger(__name__)

OUTPUT_DIR = ".sequelizer"


def split_into_chunks(data, num_chunks):
    """Split data into num_chunks chunks."""
    avg = len(data) // num_chunks
    remainder = len(data) % num_chunks
    chunks = []
    start = 0

    for i in range(num_chunks):
        end = start + avg + (1 if i < remainder else 0)
        chunks.append(data[start:end])
        start = end

    return chunks


def set_cpu_affinity(pid=None):
    """
    On Ubuntu 24.04, the same two affinity cores get assigned to every forked process (see: https://github.com/ipython/ipython/issues/840)
    ... this function assign's a unique core based on a process' ID.
    """
    import platform

    if platform.system() != "Linux" or "SLURM_JOB_ID" in os.environ:
        return

    if pid is None:
        pid = os.getpid()

    num_cores = multiprocessing.cpu_count()
    assigned_core = pid % num_cores

    try:
        os.sched_setaffinity(pid, {assigned_core})
        logger.debug(
            f"Successfully set CPU affinity for process {pid} to use core {assigned_core}."
        )
    except Exception as e:
        logger.error(
            f"Failed to set CPU affinitset_cpu_affinityy for process {pid}: {str(e)}"
        )


def setup_unique_scala_cache():
    # create unique scala repl cache, otherwise we get weird race conditions
    current_dir = os.getcwd()
    unique_cache_dir = os.path.join(current_dir, f".cache/scala-repl-pp-{uuid.uuid4()}")
    os.makedirs(unique_cache_dir, exist_ok=True)
    env = os.environ.copy()
    env["SCALA_REPL_PP_CACHE_DIR"] = unique_cache_dir
    return env


def define_all_preprocessors_checks(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    # define everything, otherwise the code is ignored
    content = re.sub(
        r"(#\s*(ifdef)\s+(\w+\S))", r"#ifdef __STDC__", content, flags=re.MULTILINE
    )

    content = re.sub(
        r"(#\s*(ifdef)\s+(\w+\S))",
        r"#ifndef __FLAT_EARTH__",
        content,
        flags=re.MULTILINE,
    )

    content = re.sub(r"static INLINE", r"static inline", content, flags=re.MULTILINE)

    with open(file_path, "w") as file:
        file.write(content)


def add_breaks_after_preprocessor_directives(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    # no argument directives
    content = re.sub(r"(#\s*(endif|else)\b.*?)", r"\n\1\n", content, flags=re.MULTILINE)

    # single argument directives
    content = re.sub(
        r"(#\s*(ifdef|ifndef|undef)\s+\w+)(\S)", r"\n\1\n", content, flags=re.MULTILINE
    )

    # define everything, otherwise the code is ignored
    content = re.sub(
        r"(#\s*(ifdef)\s+(\w+\S))", r"#ifdef __STDC__", content, flags=re.MULTILINE
    )

    content = re.sub(
        r"(#\s*(ifdef)\s+(\w+\S))",
        r"#ifndef __FLAT_EARTH__",
        content,
        flags=re.MULTILINE,
    )

    content = re.sub(r"static INLINE", r"static inline", content, flags=re.MULTILINE)

    # if directive with arithmetic argument
    content = re.sub(
        r"(#\s*if\s+)(\S+(?:\s*(?:&&|&|\|\||\||<|>|!=|==|<=|>=|\+|-|\*|\/)\s\S+)*)",
        r"\n\g<0>\n",
        content,
        flags=re.MULTILINE,
    )

    # yes..
    content = re.sub(
        r"#define\s+(\w+)(\s*(\((?:[^)(]|\((?:[^)(]|\((?:[^)(]|\([^)(]*\))*\))*\))*\)|\w+[\s]))?",
        r"\n\g<0>\n",
        content,
        flags=re.DOTALL,
    )

    with open(file_path, "w") as file:
        file.write(content)


def move_vul_tag(file_path):
    """
    Moves trailing <vul-start> tags to the next line while preserving the code as is.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    if len(lines) == 0:
        return

    def is_statement(code_line):
        s_line = (
            code_line.replace("/*<vul-start>*/", "")
            .replace("/*<vul-end>*/", "")
            .strip()
        )

        return s_line and s_line not in [
            "}",
            "{",
            ";",
            "break;",
            "continue;",
            "return;",
        ]

    def get_leading_spaces(line):
        return line[: len(line) - len(line.lstrip())]

    modified_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        leading_spaces = get_leading_spaces(line)

        parts = re.split(r"(/\*<vul-start>\*/)", line)

        if len(parts) > 1:  # vul-start tags in this line
            code_part = parts[0]
            tags = parts[1:]

            # the code part (everything before the first vul-start tag)
            modified_lines.append(leading_spaces + code_part.lstrip())

            for j in range(0, len(tags), 2):
                vul_start_tag = tags[j]
                code_after_tag = tags[j + 1] if j + 1 < len(tags) else ""

                if code_after_tag.strip():
                    # code after the tag on the same line, keep it there
                    modified_lines[-1] += vul_start_tag + code_after_tag.rstrip()
                elif i + 1 < len(lines) and is_statement(lines[i + 1]):
                    # next line is a statement, move the tag to the beginning of that line
                    next_line = lines[i + 1].rstrip()
                    next_leading_spaces = get_leading_spaces(next_line)
                    lines[i + 1] = (
                        next_leading_spaces + vul_start_tag + next_line.lstrip()
                    )
                else:
                    # no next line or it's not a statement, keep the tag on the current line
                    modified_lines[-1] += vul_start_tag
        else:
            modified_lines.append(line)

        i += 1

    if lines[-1].endswith("\n"):
        modified_lines.append("")

    with open(file_path, "w") as file:
        file.write("\n".join(modified_lines))
