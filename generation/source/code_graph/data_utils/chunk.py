import logging
from math import floor

import transformers
from transformers import AutoTokenizer

# constants
logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_error()


class GraphT5TokenEstimator:
    """
    Estimates the token count for a possible chunk based on the dataset structure of GraphT5.
    """

    def __init__(self, tokenizer, prefixes=None, ignore_tokens=None):
        self.code_tokenizer = tokenizer
        self.prefixes = prefixes
        self.ignore_tokens = ignore_tokens

    def estimate_tokens(self, text):
        if self.prefixes:
            text = " ".join(self.prefixes) + " " + text

        tokens = self.code_tokenizer.encode(text)
        if self.ignore_tokens:
            ignore_token_ids = set()
            for token in self.ignore_tokens:
                ignore_token_ids.update(
                    self.code_tokenizer.encode(token, add_special_tokens=False)
                )
            tokens = [t for t in tokens if t not in ignore_token_ids]

        return len(tokens)


class DatasetChunker:
    """
    Partitions input segments into chunks adhering to the maximum token length per segment.
    """

    def __init__(
        self,
        tokenizer_name="Salesforce/codet5-base",
        tokens=None,
        ignore_tokens=None,
    ):
        self.tokens = tokens
        self.ignore_tokens = ignore_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, local_files_only=True, use_fast=True
        )

        if tokens:
            self.tokenizer.add_tokens(tokens)

    def chunk_title(
        self,
        title,
        text,
        title_prefix,
        text_prefix,
        max_tokens=512,
    ):
        segments = []

        estimator = GraphT5TokenEstimator(
            tokenizer=self.tokenizer,
            prefixes=[title_prefix, text_prefix],
            ignore_tokens=self.ignore_tokens,
        )
        title_tokens = self.tokenizer.encode(title, add_special_tokens=False)
        start = 0
        step = 1

        while start < len(title_tokens):
            end = start
            while end < len(title_tokens):
                chunk_title_tokens = title_tokens[start : end + step]
                chunk_title = self.tokenizer.decode(
                    chunk_title_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

                token_count = estimator.estimate_tokens(chunk_title)

                if token_count > max_tokens:
                    break

                end += step

                step = max(floor((max_tokens - token_count) / 2), 1)

            chunk_title_tokens = title_tokens[start:end]
            chunk_title = self.tokenizer.decode(
                chunk_title_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            segments.append({"title": chunk_title, "text": text})

            start = end

        return segments

    def chunk_title_with_vulnerable_trim(
        self,
        title,
        text,
        title_prefix,
        text_prefix,
        max_tokens=512,
        trim_threshold=1024,
        token_limit=2560,
        padding=512,
    ):
        """
        Combines chunk_text with trim_function_to_vulnerable_chunks.

        Parameters:
        title (str): The vulnerable input sequence.
        text (str): The fixed sequence.
        title_prefix (str): The prefix for the title.
        text_prefix (str): The prefix for the text.
        max_tokens (int): Maximum number of tokens for each chunk (e.g.,512 tokens).
        trim_threshold (int): Maximum number of tokens a function may have before trimming.
        token_limit (int): Maximum number of tokens allowed for the entire sequence (default is 2560 tokens).
        padding (int): Number of additional tokens to include around each vulnerable section for context (default is 512 tokens).

        Returns:
        list of dict: List of dictionaries containing chunked text with title and text.
        """
        assert not trim_threshold or token_limit >= trim_threshold

        title_tokens = self.tokenizer.encode(title, add_special_tokens=False)
        text_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        total_tokens = len(title_tokens) + len(text_tokens)

        if trim_threshold and total_tokens > trim_threshold:
            title = self.trim_function_to_vulnerable_chunks(
                title, token_limit=token_limit, padding=padding
            )

        return self.chunk_title(title, text, title_prefix, text_prefix, max_tokens)

    def trim_function_to_vulnerable_chunks(
        self, function_code, token_limit=2560, padding=512
    ):
        """
        Trims the function to include chunks around vulnerable tags, concatenated with <vul-sep>,
        ensuring the total number of tokens does not exceed the specified token limit.

        Tries to include the function's signature if possible.

        Parameters:
        function_code (str): The function code as string.
        token_limit (int): Maximum number of tokens allowed (default is 2560 tokens = 5 segments).
        padding (int): Number of additional tokens to include around each vulnerable section for context (default is 512).

        Returns:
        str: Trimmed function code.
        """
        tokens = self.tokenizer.encode(function_code, add_special_tokens=False)
        vul_start_token = self.tokenizer.encode(
            "<vul-start>", add_special_tokens=False
        )[0]
        vul_end_token = self.tokenizer.encode("<vul-end>", add_special_tokens=False)[0]
        vul_sep_token = self.tokenizer.encode("<vul-sep>", add_special_tokens=False)

        def is_semicolon(token):
            return ";" in self.tokenizer.decode(
                token
            )  # ugly hack, but there are different tokens for semicolon (whitespace)

        chunks = []
        last_end_index = -1

        # this assumes that 4*padding will capture all <vul-start>...<vul-end> pairs.
        # will raise warning later if this is not the case
        for i, token in enumerate(tokens):
            if token == vul_start_token or token == vul_end_token:
                # whenever possible, try to build consecutive chunks
                start_index = max(last_end_index + 1, i - padding)
                end_index = min(len(tokens), i + padding + 1)

                if padding:
                    if start_index != last_end_index + 1:
                        # adjust start_index to be after the first semicolon token within the padding
                        for j in range(start_index, min(i, end_index)):
                            if is_semicolon(tokens[j]):
                                start_index = j
                                break
                    if end_index != len(tokens):
                        # adjust end_index to be the last semicolon token within the padding
                        last_semicolon_index = None
                        for j in range(max(i, start_index), end_index):
                            if is_semicolon(tokens[j]):
                                last_semicolon_index = j
                        if last_semicolon_index is not None:
                            end_index = last_semicolon_index

                chunk = tokens[start_index:end_index]
                chunks.append((chunk, start_index, end_index))

                last_end_index = end_index - 1

        # merge chunks with <vul-sep> and ensure it fits within the token limit
        merged_chunks = []
        token_count = 0
        previous_end_index = -1

        for j, (chunk, start_index, end_index) in enumerate(chunks):
            chunk_token_count = len(chunk)

            if self.ignore_tokens:
                ignore_token_ids = set()
                for token in self.ignore_tokens:
                    ignore_token_ids.update(
                        self.tokenizer.encode(token, add_special_tokens=False)
                    )
                chunk_token_count = len([t for t in chunk if t not in ignore_token_ids])

            if token_count + chunk_token_count + len(vul_sep_token) > token_limit:
                logger.error(
                    f"Could not fit all vulnerable sections for ''{function_code[:60]}''. "
                    f"Either increase token_limit or ignore."
                )
                break
                # raise Exception(
                #     f"Could not fit all vulnerable sections for {function_code}"
                # )

            if merged_chunks and previous_end_index != start_index - 1:
                merged_chunks.extend(vul_sep_token)
                token_count += len(vul_sep_token)

            merged_chunks.extend(chunk)
            token_count += chunk_token_count
            previous_end_index = end_index - 1

        # always include the function signature if it's not already included
        function_signature_tokens = tokens[: tokens.index(vul_start_token)]
        signature_token_count = len(function_signature_tokens)

        if self.ignore_tokens:
            ignore_token_ids = set()
            for token in self.ignore_tokens:
                ignore_token_ids.update(
                    self.tokenizer.encode(token, add_special_tokens=False)
                )
            signature_token_count = len(
                [t for t in function_signature_tokens if t not in ignore_token_ids]
            )

        assert merged_chunks  # that would mean that there are super long functions without vul tags in our dataset
        if merged_chunks[:signature_token_count] != function_signature_tokens:
            if token_count + signature_token_count <= token_limit:
                merged_chunks = function_signature_tokens + merged_chunks
            else:
                logger.warning(
                    f"Could not add function signature {function_code[:100]}"
                )

        return self.tokenizer.decode(merged_chunks, skip_special_tokens=True)
