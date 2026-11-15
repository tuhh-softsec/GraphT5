import json
import os
import re
import traceback
from enum import Enum

from orjson import orjson
from tqdm import tqdm
from transformers import AutoTokenizer

# constants
MAX_SEGMENT_LENGTH = 512

LINE_DELIMITER = "<line-end>"

GRAPHT5_TOKENS = [
    "<vul-start>",
    "<vul-end>",
    "<vul-sep>",
    "<blank>",
    "<node>",
    LINE_DELIMITER,
    "</node>",
    "<space>",
    "<blank>",
    "<null>",
    "<AST>",
    "<DDG>",
    "</DDG>",
    "<CDG>",
    "<CFG>",
    "<REF>",
    "</REF>",
]


class SegmentType(Enum):
    VULNERABLE_AST_INPUT = (
        "Vulnerable AST Input",
        r"CWE-\d+ Code Input AST Vulnerable Code Is:(?: CWE-\d+)?",
        r"CWE-\d+ Fixed Code Lines Are:",
    )
    VULNERABLE_VDG_INPUT = (
        "Vulnerable VDG Input",
        r"CWE-\d+ Code Input VDG Vulnerable Code Is:(?: CWE-\d+)?",
        r"CWE-\d+ Fixed Code Lines Are:",
    )
    VULNERABLE_CODE_INPUT = (
        "Vulnerable Code Input",
        r"CWE-\d+ Code Input Vulnerable Code Is:(?: CWE-\d+)?",
        r"CWE-\d+ Fixed Code Lines Are:",
    )
    VULNERABLE_CWE_CODE = (
        "Vulnerable CWE Code",
        r"CWE-\d+ Vulnerable Code Is:",
        r"CWE-\d+ Fixed Code Lines Are:",
    )
    CWE_DESCRIPTION = (
        "Vulnerable CWE Description",
        r"CWE-\d+ Name:",
        r"CWE-\d+ Fixed Code Lines Are:",
    )
    UNKNOWN_CWE = (
        "Unknown",
        r"CWE-\d+ Unknown CWE type and no descriptions",
        r"CWE-\d+ Fixed Code Lines Are:",
    )

    def instantiate_title(self, number):
        return self.__instantiate(self.value[1], number)

    def instantiate_text(self, number):
        return self.__instantiate(self.value[2], number)

    def __instantiate(self, template, number):
        return template.replace(r"(?: CWE-\d+)?", "").replace(r"\d+", str(number))


def load_json_file(file_path):
    try:
        with open(file_path, "r") as file:
            data = orjson.loads(file.read())
        return data
    except orjson.JSONDecodeError:
        data = []
        with open(file_path, "r") as file:
            for sample in file:
                data.append(orjson.loads(sample))
        return data
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def save_dataset_to_file(file_path, data, indent=4):
    try:
        directory = os.path.dirname(file_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, "w") as file:
            json.dump(data, file, indent=indent)
    except Exception as e:
        print(f"An unexpected error occurred while trying to save dataset: {e}")


def extract_vulnerable_code_from_java(data, key="vulnerable_func"):
    vulnerable_code_inputs = []
    for k, item in tqdm(
        enumerate(data), desc="Loading sequences and reassembling... ", total=len(data)
    ):
        wrapper = f"class XmlFactory {{ {item[key]} }}"

        cwe_id = item["cwe_id"] if "cwe_id" in item else "CWE-000"
        vulnerable_code_inputs.append((k, cwe_id, wrapper, item["target"]))

    return vulnerable_code_inputs


def extract_segments(data, segment_type: SegmentType, ignore_zero=False):
    vulnerable_code_inputs = []

    for k, item in tqdm(
        enumerate(data), desc="Loading sequences and reassembling... ", total=len(data)
    ):
        try:
            # target = (item["target"].split(":", 1)[1:][0]).lstrip()
            target = item["target"]
            cwe_id = item["cwe_id"]

            vulnerable_code = item["passages"].replace(LINE_DELIMITER, "\n")

            vulnerable_code_inputs.append((k, cwe_id, vulnerable_code, target))
        except KeyError as e:
            print(f"Missing key {e} in item: {item}")
        except ValueError as e:
            print(f"Value error while parsing: \n{e}")

    return vulnerable_code_inputs


def inject_segments(
    data,
    segments,
    segment_type: SegmentType,
    replace=False,
    node_link=False,
    unique_index=300000,
):

    tokenizer = AutoTokenizer.from_pretrained(
        "Salesforce/codet5-base", use_fast=True, local_files_only=False
    )
    tokenizer.add_tokens(GRAPHT5_TOKENS)

    if len(data) != len(segments):
        raise ValueError(
            f"The lengths of dataset items and segments {len(data), len(segments)} do not match."
        )

    for k, (segment, segment_list) in enumerate(segments):
        try:
            item = data[k]

            if node_link:
                item["vdg"] = segment_list
            else:
                cwe_id = segment[1].split(" ")[0].split("-")[1]
                segment_type.instantiate_title(cwe_id)

                # there is a repeated duplication in VulMaster's dataset
                # each segment array also contains the question as id 0.
                # these indices are purely cosmetic, no inherent value
                segment_dicts = [
                    {
                        "id": str(unique_index + int(segment[0]) * 100 + idx),
                        "title": passage["title"],
                        "text": passage[
                            "text"
                        ],  # {' ' + answer[0] if answer else ''}",
                    }
                    for idx, passage in enumerate(segment_list)
                ]

                # make sure that all segments fit
                for compiled_segment in segment_dicts:
                    tk_length = len(
                        tokenizer.encode(
                            compiled_segment["title"] + " " + compiled_segment["text"]
                        )
                    )

                    if tk_length > MAX_SEGMENT_LENGTH:
                        raise Exception(
                            f"Uff! Behold! One of the {segment_type.value[0]} segments exceeds the max. token length ({tk_length}/{MAX_SEGMENT_LENGTH})!"
                        )

                if replace and segment_dicts:
                    ctxs = item["ctxs"]
                    start_index, filtered_seqs = _remove_seq_from(ctxs, segment_type)
                    new_ctxs = (
                        filtered_seqs[:start_index]
                        + segment_dicts
                        + filtered_seqs[start_index:]
                    )

                    item["ctxs"] = new_ctxs
                else:
                    item["ctxs"].extend(segment_dicts)

        except IndexError as e:
            print(f"Can not find matching item in dataset for index {k}: {e}")
        except KeyError as e:
            print(f"Cannot find passages: {e} in item: {data[k]}")

    return data


# leaving this here, this was the previous approach
def tokenize_partition_and_detokenize(code, tokenizer, max_length=512):
    tokens = tokenizer.encode(
        code,
        add_special_tokens=False,
    )

    token_segments = []
    current_segment = []
    current_length = 0

    for token in tokens:
        current_segment.append(token)
        current_length += 1

        if current_length >= max_length:
            token_segments.append(current_segment)
            current_segment = []
            current_length = 0

    if current_segment:
        token_segments.append(current_segment)

    code_segments = [
        tokenizer.decode(segment, clean_up_tokenization_spaces=True)
        for segment in token_segments
    ]

    return code_segments


def _remove_seq_from(ctxs, segment_type: SegmentType):
    start_index = -1
    filtered_seqs = []

    for k, seq in enumerate(ctxs):
        if re.match(segment_type.value[1], seq.get("title", "")):
            if start_index == -1:
                start_index = k
        else:
            filtered_seqs.append(seq)

    start_index = start_index if start_index >= 0 else len(ctxs)

    return start_index, filtered_seqs
