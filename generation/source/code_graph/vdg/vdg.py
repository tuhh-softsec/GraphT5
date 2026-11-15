import logging
import multiprocessing
import os
import queue
from functools import partial
from multiprocessing import Queue
from typing import Optional

import tqdm
from code_graph.data import GRAPHT5_TOKENS, SegmentType
from code_graph.data_utils.chunk import DatasetChunker
from code_graph.joern.joern import (
    generate_cpgs,
    parse_joern_node_link,
    parse_joern_vdg,
    run_joern,
)
from code_graph.joern.vdg.vdg_generator import VDGGenerator
from code_graph.util import (
    OUTPUT_DIR,
    set_cpu_affinity,
    setup_unique_scala_cache,
    split_into_chunks,
)
from code_graph.vdg.walk_vdg import structured_vdg_traversal

logger = logging.getLogger(__name__)

# constants
helpers_dir = os.path.dirname(__file__)


def single_chunk_vdg_node_link_worker(
    segments,
    cpg_queue: Queue,
    vdg_queue: Queue,
    stats_dir: str = None,
    lang: Optional[str] = "cpp",
    verbose=False,
):
    # os.system("taskset -ap 0xff %d" % os.getpid())
    set_cpu_affinity()

    # Joern comes with its own custom scala repl.
    # which is apparently not thread-safe.
    setup_unique_scala_cache()

    # dataset_chunker = DatasetChunker(tokenizer_name="Salesforce/codet5-base", tokens=VULMASTER_TOKENS)

    sources = []
    stats_base_paths = []
    segment_ids = []

    # Prepare the sources and stats_base_paths
    for segment in segments:
        source = (
            "".join(segment[2])
            .replace("<vul-start>", "/*<vul-start>*/")
            .replace("<vul-end>", "/*<vul-end>*/")
            .replace("<line-end>", "\n")
        )
        sources.append(source)

        if stats_dir:
            stats_base_path = (
                stats_dir + f"/segment_{segment[0]}" if stats_dir else None
            )
            stats_base_paths.append(stats_base_path)

        segment_ids.append(segment[0])

    sequences = []

    try:
        # generate cpgs
        cpg_files = []
        source_files = []

        for cpg_file, source_file in generate_cpgs(sources, segment_ids, lang=lang):
            cpg_files.append(cpg_file)
            source_files.append(source_file)
            cpg_queue.put(1, block=False)

        # run Joern for all files at once and handle progress
        for idx, processed_file in enumerate(
            run_joern(
                source_files,
                cpg_files,
                stats_base_paths=stats_base_paths,
                lang=lang,
                verbose=verbose,
            )
        ):
            # assert processed_file, "Joern exited with error!"
            vdg = parse_joern_node_link(processed_file)

            if not vdg:
                logger.debug(
                    f"Extracted empty VDG from code sample with id {segment_ids[idx]}."
                )

            sequences.append((segments[idx], vdg))

            # update the progress queue
            vdg_queue.put(1, block=False)
    except Exception:
        logger.exception(f"Error processing VDG:")
        raise

    return sequences


def single_chunk_vdg_worker(
    segments, cpg_queue: Queue, vdg_queue: Queue, stats_dir: str = None, verbose=False
):
    # os.system("taskset -ap 0xff %d" % os.getpid())
    set_cpu_affinity()

    # Joern comes with its own custom scala repl.
    # which is apparently not thread-safe.
    setup_unique_scala_cache()

    dataset_chunker = DatasetChunker(
        tokenizer_name="Salesforce/codet5-base", tokens=GRAPHT5_TOKENS
    )

    sources = []
    stats_base_paths = []
    segment_ids = []

    # Prepare the sources and stats_base_paths
    for segment in segments:
        source = (
            "".join(segment[2])
            .replace("<vul-start>", "/*<vul-start>*/")
            .replace("<vul-end>", "/*<vul-end>*/")
        )
        sources.append(source)

        if stats_dir:
            stats_base_path = (
                stats_dir + f"/segment_{segment[0]}" if stats_dir else None
            )
            stats_base_paths.append(stats_base_path)

        segment_ids.append(segment[0])

    sequences = []

    try:
        # generate cpgs
        cpg_files = []
        source_files = []

        for cpg_file, source_file in generate_cpgs(sources, segment_ids):
            cpg_files.append(cpg_file)
            source_files.append(source_file)
            cpg_queue.put(1, block=False)

        # run Joern for all files at once and handle progress
        for idx, processed_file in enumerate(
            run_joern(
                source_files,
                cpg_files,
                stats_base_paths=stats_base_paths,
                verbose=verbose,
            )
        ):
            assert processed_file, "Joern exited with error!"
            vdg = parse_joern_vdg(processed_file)

            if not vdg:
                logger.debug(
                    f"Extracted empty VDG from code sample with id {segment_ids[idx]}."
                )

            seq = (
                structured_vdg_traversal(vdg)
                .replace("/*<vul-start>*/", "<vul-start>")
                .replace("/*<vul-end>*/", "<vul-end>")
            )

            cwe_id = segments[idx][1].split(" ")[0].split("-")[1]
            title_prefix = SegmentType.VULNERABLE_VDG_INPUT.instantiate_title(cwe_id)

            text_prefix = f"CWE-{cwe_id} Fixed Code Lines are:"

            chunked_segments = dataset_chunker.chunk_title(
                title=seq,
                text="",
                title_prefix=title_prefix,
                text_prefix=text_prefix,
                max_tokens=512,
            )
            sequences.append((segments[idx], chunked_segments))

            # update the progress queue
            vdg_queue.put(1, block=False)
    except Exception:
        logger.exception("An unexpected error occurred while trying to generate VDG:")

    return sequences


def generate_vdg_for_segments(
    segments,
    num_proc=8,
    node_link=False,
    stats_dir=None,
    lang: Optional[str] = "cpp",
    verbose=False,
):
    assert num_proc >= 1

    filtered_segments = []
    skipped_indices = []

    segment_by_id = {segment[0]: (idx, segment) for idx, segment in enumerate(segments)}
    existing_vdg_files = {}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with os.scandir(OUTPUT_DIR) as dir_entries:
        for entry in dir_entries:
            if entry.name.endswith("_node_link.json"):
                segment_id = int(entry.name.split("_node_link.json")[0])
                existing_vdg_files[segment_id] = entry.path

    with open("process.log", "a", encoding="utf-8") as logfile:
        for idx, segment in enumerate(
            tqdm.tqdm(
                segments,
                desc="Resuming previous generation..",
                unit="VDG",
                ascii=True,
            )
        ):
            segment_id = segment[0]

            if segment_id in existing_vdg_files:
                vdg_file = existing_vdg_files[segment_id]

                if existing_vdg := VDGGenerator.validate_vdg(vdg_file):
                    skipped_indices.append((idx, segment, existing_vdg))
                    verbose and logger.debug(
                        f"Skipping existing VDG for segment {segment_id}"
                    )
                    continue
                else:
                    logger.warning(
                        f"Found corrupted VDG file for segment {segment_id}, will regenerate"
                    )
                    os.unlink(vdg_file)

            filtered_segments.append(segment)

    if not filtered_segments:
        logger.info(
            f"All VDGs already have been processed. Remove {OUTPUT_DIR} if you want to regenerate.."
        )
        return [(seg[1], seg[2]) for seg in skipped_indices]

    if skipped_indices:
        logger.info(f"Skipping {len(skipped_indices)} already processed VDGs.")

    chunks = split_into_chunks(filtered_segments, num_proc)

    manager = multiprocessing.Manager()
    cpg_queue = manager.Queue()
    vdg_queue = manager.Queue()

    pool = multiprocessing.Pool(processes=num_proc)

    vdgs = []
    total_segments = len(filtered_segments)

    progress_updater = multiprocessing.Process(
        target=_update_progress,
        args=(cpg_queue, vdg_queue, total_segments),
    )
    progress_updater.start()

    worker_function = partial(
        single_chunk_vdg_node_link_worker if node_link else single_chunk_vdg_worker,
        cpg_queue=cpg_queue,
        vdg_queue=vdg_queue,
        stats_dir=stats_dir,
        lang=lang,
        verbose=verbose,
    )

    for result in pool.imap(worker_function, chunks):
        vdgs.extend(result)

    pool.close()
    pool.join()
    _finish_progress(progress_updater)

    final_results = []
    new_vdg_idx = 0
    skipped_idx = 0

    for i in range(len(segments)):
        if skipped_idx < len(skipped_indices) and skipped_indices[skipped_idx][0] == i:
            final_results.append(
                (skipped_indices[skipped_idx][1], skipped_indices[skipped_idx][2])
            )
            skipped_idx += 1
        else:
            if new_vdg_idx < len(vdgs):
                final_results.append(vdgs[new_vdg_idx])
                new_vdg_idx += 1

    return final_results


def _finish_progress(progress_updater, timeout=2):
    progress_updater.join(timeout)
    if progress_updater.is_alive():
        logger.warning("Stopping incomplete VDG progress tracking.")
        progress_updater.terminate()
        progress_updater.join()


def _update_progress(cpg_queue, vdg_queue, total_items):
    with (
        open("process.log", "a") as log_file,
        tqdm.trange(
            total_items,
            desc="Generating CPGs from source",
            unit="CPG",
            smoothing=0.1,
            position=0,
            leave=True,
            file=log_file,
            ascii=True,
        ) as cpg_pbar,
    ):
        remaining_cpgs = total_items
        while remaining_cpgs > 0:
            try:
                cpg_queue.get(timeout=1.0)
                cpg_pbar.update(1)
                remaining_cpgs -= 1
            except queue.Empty:
                continue

        cpg_pbar.n = total_items
        cpg_pbar.refresh()

    logger.info("done.. firing up VMs, this may take a while...")

    with (
        open("process.log", "a") as log_file,
        tqdm.trange(
            total_items,
            desc="Generating VDGs from CPGs",
            unit="VDG",
            smoothing=0.1,
            position=1,
            leave=True,
            file=log_file,
            ascii=True,
        ) as vdg_pbar,
    ):
        remaining_vdgs = total_items
        while remaining_vdgs > 0:
            try:
                vdg_queue.get(timeout=1.0)
                vdg_pbar.update(1)
                remaining_vdgs -= 1
            except queue.Empty:
                continue

        vdg_pbar.n = total_items
        vdg_pbar.refresh()


def _find_vulnerable_lines(comments):
    vulnerable_ranges = []
    current_start = None

    for _, row in comments.iterrows():
        if "<vul-start>" in row["code"]:
            current_start = row["line_number"]
        elif "<vul-end>" in row["code"] and current_start is not None:
            current_end = row["line_number"]
            if current_start <= current_end:
                vulnerable_ranges.extend(range(current_start, current_end + 1))
            current_start = None

    return vulnerable_ranges
