import argparse
import ast
import difflib
import json
import os
import time

from code_graph.data_utils.mod.source_tokenizer import tokenize_source


def convert_to_proper_json(line):
    try:
        python_dict = ast.literal_eval(line)
        return json.dumps(python_dict)
    except Exception as e:
        raise ValueError(f"Failed to convert line to proper JSON: {str(e)}")


def get_token_pair_diff(pre_token_per_line, post_token_per_line, num_tokens=3):
    """
    Generates patch modifiacation instructions.
    """
    try:
        diff = list(
            difflib.unified_diff(
                pre_token_per_line,
                post_token_per_line,
                fromfile="buggy",
                tofile="fixed",
                n=1000000,
            )
        )

        if not diff:
            return None, None

        state = 0
        pre_tokens = ["<null>"] * num_tokens
        post_tokens = ["<null>"] * num_tokens
        new_tokens = []
        tgt = ""

        src = ""
        src_line = ""
        bugtag = False

        for t in diff:
            t = t.replace("\n", "")
            token = t[1:]

            if t.startswith("--- ") or t.startswith("+++ ") or t.startswith("@@ "):
                state += 1
                continue

            if state < 3:
                continue

            if t.startswith(" "):
                if t != " /*LINEEND*/" and t != " /*LINECONTINUATION*/":
                    src_line += token + " "
                else:
                    delimiter = "\\\n" if t == " /*LINECONTINUATION*/" else "\n"

                    if bugtag:
                        src += "<vul-start> " + src_line + "<vul-end> " + delimiter
                        bugtag = False
                        src_line = ""
                        continue
                    else:
                        src += src_line + delimiter
                        src_line = ""
                        continue

                if state == 3:
                    pre_tokens = pre_tokens[1:num_tokens] + [token]
                elif state % 100 == num_tokens - 1:
                    post_tokens = post_tokens[1:] + [token]

                    if state >= 300:
                        tgt += f'<mod-start> {" ".join(pre_tokens)} {" ".join(new_tokens)} '
                    elif state >= 200:
                        tgt += f'<mod-start> {" ".join(pre_tokens)} {" ".join(new_tokens)} <mod-end> {" ".join(post_tokens)} '
                    elif state >= 100:
                        tgt += f'<mod-start> {" ".join(pre_tokens)} <mod-end> {" ".join(post_tokens)} '
                    state = 3
                    pre_tokens = post_tokens
                else:
                    state += 1
                    post_tokens = post_tokens[1:num_tokens] + [token]

            elif t.startswith("-"):
                if t != "-/*LINEEND*/" and t != "-/*LINECONTINUATION*/":
                    src_line += token + " "
                else:
                    delimiter = "\\\n" if t == "-/*LINECONTINUATION*/" else "\n"
                    if bugtag:
                        src += "<vul-start> " + src_line + "<vul-end> " + delimiter
                        bugtag = False
                        src_line = ""
                        continue
                    else:
                        src += src_line + delimiter
                        src_line = ""
                        continue

                if state == 3:
                    bugtag = True
                    state = 100
                    new_tokens = []
                elif state >= 300:
                    new_tokens += post_tokens[num_tokens - (state % 100) : num_tokens]
                    state = 200
                elif state >= 200:
                    new_tokens += post_tokens[num_tokens - (state % 100) : num_tokens]
                    state = 200
                elif state > 100:
                    new_tokens += post_tokens[num_tokens - (state % 100) : num_tokens]
                    state = 200
                post_tokens = ["<null>"] * num_tokens

            elif t.startswith("+"):
                if t == "+/*LINEEND*/" or t == "+/*LINECONTINUATION*/":
                    continue

                if state == 3:
                    bugtag = True
                    state = 300
                    new_tokens = [token]
                elif state >= 300:
                    new_tokens += post_tokens[
                        num_tokens - (state % 100) : num_tokens
                    ] + [token]

                    if state > 300:
                        state = 200

                elif state >= 200:
                    new_tokens += post_tokens[
                        num_tokens - (state % 100) : num_tokens
                    ] + [token]
                    state = 200
                elif state >= 100:
                    new_tokens += post_tokens[
                        num_tokens - (state % 100) : num_tokens
                    ] + [token]
                    state = 200
                post_tokens = ["<null>"] * num_tokens

        if src_line:
            if bugtag:
                src += "<vul-start> " + src_line + "<vul-end> "
            else:
                src += src_line

        post_tokens = (
            post_tokens[num_tokens - (state % 100) : num_tokens]
            + post_tokens[0 : num_tokens - (state % 100)]
        )

        if state >= 300:
            tgt += f'<mod-start> {" ".join(pre_tokens)} {" ".join(new_tokens)} '
        elif state >= 200:
            tgt += f'<mod-start> {" ".join(pre_tokens)} {" ".join(new_tokens)} <mod-end> {" ".join(post_tokens)} '
        elif state >= 100:
            tgt += (
                f'<mod-start> {" ".join(pre_tokens)} <mod-end> {" ".join(post_tokens)} '
            )

        return src.strip(), tgt.strip()

    except Exception as e:
        print(f"Error in get_token_pair_diff: {str(e)}")
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Process JSON files containing code pairs and generate token pair diffs.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input JSON file containing code pairs",
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Path to the output text file (default: diff_output_[timestamp].txt)",
    )

    parser.add_argument(
        "--json_output",
        "-j",
        help="Path to the output JSON file (default: diff_dataset_[timestamp].json)",
    )

    parser.add_argument(
        "--num-tokens",
        "-n",
        type=int,
        default=3,
        help="Number of context tokens to use (default: 3)",
    )

    parser.add_argument(
        "--pre-key",
        "--pre",
        default="pre",
        help="JSON key for pre/buggy code (default: 'pre')",
    )

    parser.add_argument(
        "--post-key",
        "--post",
        default="post",
        help="JSON key for post/fixed code (default: 'post')",
    )

    parser.add_argument(
        "--id-key", "--id", default="id", help="JSON key for entry ID (default: 'id')"
    )

    parser.add_argument(
        "--lang",
        default="java",
        help="Language of source code.",
    )

    parser.add_argument(
        "--trim-prefix",
        help="Optional prefix to trim from pre/post code (e.g., 'class GraphT5 {')",
    )

    parser.add_argument(
        "--trim-suffix", help="Optional suffix to trim from pre/post code (e.g., '}')"
    )

    parser.add_argument(
        "--jsonl",
        action="store_true",
        default=False,
        help="Source is JSON line file.",
    )

    parser.add_argument(
        "--source-key",
        default="passages",
        help="Key for source code in output (default: 'passages')",
    )

    parser.add_argument(
        "--target-key",
        default="target",
        help="Key for target code in output (default: 'target')",
    )

    parser.add_argument(
        "--flat",
        action="store_true",
        help="Input is a flat array of objects instead of a dictionary",
    )

    parser.add_argument(
        "--from_method_changes",
        action="store_true",
        help="Whether the input file is an array of method changes or already preprocessed",
    )

    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if not args.output:
        args.output = f"diff_output_{timestamp}.txt"

    if not args.json_output:
        args.json_output = f"diff_dataset_{timestamp}.json"

    try:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_json_dir = os.path.dirname(args.json_output)
        if output_json_dir and not os.path.exists(output_json_dir):
            os.makedirs(output_json_dir)

        data = []
        with open(args.input, "r", encoding="utf-8") as f:
            if args.jsonl:
                for line in f.readlines():
                    data.append(json.loads(line))
            else:
                data = json.load(f)

        dataset = process_json_data(data, args)

        if len(dataset) % 2 != 0:
            del dataset[-1]

        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)

        print(f"Successfully processed {len(dataset)} samples.")
        print(f"Text output written to: {args.output}")
        print(f"JSON dataset written to: {args.json_output}")

    except FileNotFoundError:
        print(f"Input file {args.input} not found")
    except json.JSONDecodeError:
        import traceback

        traceback.print_exc()
        print(f"Input file {args.input} is not a valid JSON file")
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"An error occurred: {str(e)}")


def process_json_data(data, args):
    """
    Process the JSON data to extract code pairs and generate token pair diffs.

    Args:
        data: The loaded JSON data
        args: Command-line arguments

    Returns:
        List of processed entries with diffs
    """
    import traceback

    from tqdm import tqdm

    dataset = []

    with open(args.output, "w", encoding="utf-8") as f_out:
        if args.flat:
            items = [(i, item) for i, item in enumerate(data)]
        else:
            items = [(key, item) for key, item in data.items()]

        if args.from_method_changes:
            extracted_functions = []
            id = 0

            for _, item in items:
                method_changes = item["method_changes"]

                # if len(method_changes) != 1:
                #     continue

                for method in method_changes:
                    # method = method_changes[0]
                    pre_source = method["pre_source"].strip()
                    post_source = method["post_source"].strip()

                    if pre_source != "" and post_source != "":
                        item_obj = {
                            args.pre_key: method["pre_source"],
                            args.post_key: method["post_source"],
                            **item,
                        }

                        extracted_functions.append((id, item_obj))
                        id += 1

            items = extracted_functions

        for key, item in tqdm(items, desc="Processing entries"):
            try:
                entry_id = item.get(args.id_key, key)

                pre_code = item.get(args.pre_key)
                post_code = item.get(args.post_key)

                if pre_code.startswith("\n"):
                    pre_code = pre_code.lstrip("\n ")
                if post_code.startswith("\n"):
                    post_code = post_code.lstrip("\n ")

                # if pre_code.startswith("protected "):
                #     pre_code = pre_code.lstrip("protected ")
                # if pre_code.startswith("private "):
                #     pre_code = pre_code.lstrip("private ")
                # if pre_code.startswith("public "):
                #     pre_code = pre_code.lstrip("public ")
                #
                # if post_code.startswith("protected "):
                #     post_code = post_code.lstrip("protected ")
                # if post_code.startswith("private "):
                #     post_code = post_code.lstrip("private ")
                # if post_code.startswith("public "):
                #     post_code = post_code.lstrip("public ")

                if pre_code is None or post_code is None:
                    print(
                        f"Warning: Entry {entry_id} missing required fields ({args.pre_key} or {args.post_key}). Skipping."
                    )
                    continue

                if args.trim_prefix:
                    pre_code = pre_code.lstrip(args.trim_prefix)
                    post_code = post_code.lstrip(args.trim_prefix)

                if args.trim_suffix:
                    pre_code = pre_code.rstrip(args.trim_suffix)
                    post_code = post_code.rstrip(args.trim_suffix)

                lang = item.get("lang", args.lang)

                buggy_tokens = tokenize_source(pre_code, lang=lang, line_tags=True)
                fixed_tokens = tokenize_source(post_code, lang=lang, line_tags=True)

                source, target = get_token_pair_diff(
                    buggy_tokens, fixed_tokens, args.num_tokens
                )

                if target:
                    cwe_id = item["cwe_id"]
                    cve_id = item["cve_id"]
                    original_address = item["repo"]
                    # original_address = (
                    #     item["original_address"] + "/commit/" + item["commit_id"]
                    # )
                    # project_id = item["original_address"]
                    # commit_id = item["commit_id"]

                    entry = {
                        "id": entry_id,
                        "vul4j": item["no"],
                        args.source_key: (
                            [source] if isinstance(source, str) else source
                        ),
                        "original_func": source,  # pre_code
                        "lang": lang,
                        "cwe_id": cwe_id,
                        "cve_id": cve_id,
                        "original_address": original_address,
                        # "project_id": project_id,
                        # "commit_url": commit_id,
                        args.target_key: target,
                    }

                    dataset.append(entry)

                    f_out.write(f"=== Sample {entry_id} ===\n")
                    f_out.write(pre_code)
                    f_out.write("\nTarget: " + target + "\n\n")
                else:
                    pass
                    # no changes happened!
                    # print(entry.get("vul4j"))
                    # print(pre_code)
                    # print(post_code)

            except Exception as e:
                traceback.print_exc()
                print(f"Error processing item {key}: {str(e)}")

    grouped_by_cwe = {}
    for smp in dataset:
        grouped_by_cwe.setdefault(smp["vul4j"], []).append(smp)

    for vul4j, changes in grouped_by_cwe.items():
        if len(changes) > 1:
            print(vul4j)

    return dataset


if __name__ == "__main__":
    main()
