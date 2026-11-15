import gzip
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import msgspec
import numpy as np
import torch

from types import SimpleNamespace
from grapht5.datasets.packing import decode_packed
from grapht5.datasets.schema import Node, Sample, VDG
from grapht5.runtime.util import EDGE_TYPES

EDGE_TYPE_INDEX = {edge_type: index for index, edge_type in enumerate(EDGE_TYPES)}


def mask_empty_passages(
    attention_mask: torch.Tensor, passages: List[List[str]]
) -> torch.Tensor:
    present = torch.tensor(
        [[bool(passage.strip()) for passage in sample] for sample in passages],
        dtype=torch.bool,
        device=attention_mask.device,
    )
    return attention_mask & present.unsqueeze(-1)


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: List[Sample],
        opt: SimpleNamespace,
    ):
        self.data = data
        self.n_context = opt.n_context
        self.no_prefix = opt.no_prefix

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.data[index]
        f = "{} " + "{}"
        formatted_passages = [
            f.format(
                "" if self.no_prefix else c.prefix,
                c.text.replace("<line-end>", ""),
            )
            for c in example.passages[: self.n_context]
        ]

        target = (
            Dataset.remove_prefix(example.target) if self.no_prefix else example.target
        )

        return {
            "index": index,
            "target": target,
            "passages": formatted_passages,
            "vdg": example.vdg,
        }

    def get_example(self, index):
        return self.data[index]

    @staticmethod
    def remove_prefix(tgt):
        tgt_components = tgt[0].split(":", 1)
        if len(tgt_components) > 1:
            return tgt_components[1].removeprefix(" ")
        return tgt


class Collator:
    def __init__(
        self,
        tokenizer,
        text_maxlength: int = 512,
        answer_maxlength: int = 20,
        node_max_length: int = 16,
        max_nodes_per_graph: Optional[int] = None,
        n_context: int = 1,
        passage_dropout=None,
    ):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        self.node_max_length = node_max_length
        self.max_nodes_per_graph = max_nodes_per_graph
        self.n_context = n_context
        self.passage_dropout = passage_dropout

        self.is_training = True

    def __call__(self, batch: List[Dict]) -> Tuple:
        index = torch.tensor([ex["index"] for ex in batch])

        target_encoding = self.tokenizer.batch_encode_plus(
            [ex["target"] for ex in batch],
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            padding="max_length",
            return_tensors="pt",
            truncation=bool(self.answer_maxlength > 0),
        )
        target_mask = target_encoding["attention_mask"].bool()
        target_ids = target_encoding["input_ids"].masked_fill(~target_mask, -100)

        # passages
        # text_passages = [
        #     ex["passages"][: self.n_context]
        #     + [""] * (self.n_context - len(ex["passages"]))
        #     for ex in batch
        # ]

        text_passages = []
        for ex in batch:
            # drop entire passage for a sample with probability `passage_dropout`
            if (
                self.is_training
                and self.passage_dropout is not None
                and not is_empty_vdg(ex["vdg"])
                and np.random.random() < self.passage_dropout
            ):
                passage_prefix = ""
                if len(ex["passages"]) > 0:
                    passage_components = ex["passages"][0].split(":", 1)
                    if len(passage_components) > 0:
                        passage_prefix = passage_components[0]

                passages = [passage_prefix] + [""] * (self.n_context - 1)
                text_passages.append(passages)
            else:
                text_passages.append(
                    ex["passages"][: self.n_context]
                    + [""] * (self.n_context - len(ex["passages"]))
                )

        passage_encodings = self.tokenizer.batch_encode_plus(
            [p for passages in text_passages for p in passages],
            max_length=self.text_maxlength,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        passage_ids = passage_encodings["input_ids"].view(
            len(batch), -1, self.text_maxlength
        )
        passage_masks = (
            passage_encodings["attention_mask"]
            .view(len(batch), -1, self.text_maxlength)
            .bool()
        )
        passage_masks = mask_empty_passages(passage_masks, text_passages)

        # graphs
        nodes = self._process_graphs(batch)

        return (
            index,
            target_ids,
            target_mask,
            passage_ids,
            passage_masks,
            nodes["input_ids"],
            nodes["attention_mask"],
            nodes["roles"],
            nodes["edges"],
        )

    def eval(self):
        self.is_training = False

    def train(self):
        self.is_training = True

    def _is_empty_vdg(self, vdg):
        return is_empty_vdg(vdg)

    def _process_graphs(self, batch: List[Dict]) -> Tuple:
        processed_vdgs = [
            self._process_single_graph(example["vdg"]) for example in batch
        ]
        nodes, required_spans, node_roles, edge_lists = zip(*processed_vdgs)

        max_num_nodes = max(len(node_list) for node_list in nodes)
        node_encodings = [
            self._encode_nodes(node_list, spans, roles, max_num_nodes)
            for node_list, spans, roles in zip(nodes, required_spans, node_roles)
        ]

        max_num_edges = max(len(edge_list) for edge_list in edge_lists)
        edges = torch.full((len(batch), max_num_edges, 4), -1, dtype=torch.long)
        for index, edge_list in enumerate(edge_lists):
            if edge_list:
                edges[index, : len(edge_list)] = torch.tensor(
                    edge_list, dtype=torch.long
                )

        return {
            "input_ids": torch.stack([n["input_ids"] for n in node_encodings]),
            "attention_mask": torch.stack(
                [n["attention_mask"] for n in node_encodings]
            ),
            "roles": torch.stack([n["roles"] for n in node_encodings]),
            "edges": edges,
        }

    def _process_single_graph(self, graph: VDG) -> Tuple:
        selected_nodes = select_vdg_nodes(graph, self.max_nodes_per_graph)
        id_to_index = {node.id: idx for idx, node in enumerate(selected_nodes)}
        vulnerable_ids = get_vulnerable_node_ids(graph)
        node_roles = [int(node.id in vulnerable_ids) for node in selected_nodes]
        required_spans = defaultdict(list)
        edges = []
        seen_edges = set()
        for link in graph.links:
            if (
                link.source != link.target
                and link.source in id_to_index
                and link.target in id_to_index
            ):
                src_index = id_to_index[link.source]
                dst_index = id_to_index[link.target]
                required_spans[src_index].append(
                    (link.srcCharacters.start, link.srcCharacters.end)
                )
                required_spans[dst_index].append(
                    (link.dstCharacters.start, link.dstCharacters.end)
                )
                edge_key = (src_index, dst_index, link.type)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append(
                        (
                            src_index,
                            dst_index,
                            EDGE_TYPE_INDEX[link.type],
                            link.target - link.source,
                        )
                    )

        return (
            selected_nodes,
            required_spans,
            node_roles,
            edges,
        )

    def _encode_nodes(self, nodes, required_spans, roles, max_num_nodes: int) -> Dict:
        encoded_nodes = [
            self._encode_node(node.code, required_spans[index], bool(roles[index]))
            for index, node in enumerate(nodes)
        ]
        padding = max_num_nodes - len(nodes)
        empty_ids = torch.full(
            (self.node_max_length,),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        empty_mask = torch.zeros(self.node_max_length, dtype=torch.long)
        empty_offsets = [(0, 0)] * self.node_max_length
        encoded_ids = [encoding[0] for encoding in encoded_nodes] + [
            empty_ids
        ] * padding
        encoded_masks = [encoding[1] for encoding in encoded_nodes] + [
            empty_mask
        ] * padding
        return {
            "input_ids": (
                torch.stack(encoded_ids)
                if encoded_ids
                else torch.empty((0, self.node_max_length), dtype=torch.long)
            ),
            "attention_mask": (
                torch.stack(encoded_masks)
                if encoded_masks
                else torch.empty((0, self.node_max_length), dtype=torch.long)
            ),
            "offset_mapping": [encoding[2] for encoding in encoded_nodes]
            + [empty_offsets] * padding,
            "line_ids": torch.tensor(
                [node.id for node in nodes] + [0] * padding, dtype=torch.long
            ),
            "roles": torch.tensor(list(roles) + [0] * padding, dtype=torch.long),
        }

    def _encode_node(self, code, required_spans, vulnerable):
        spans = []
        if vulnerable:
            for marker in ("<vul-start>", "<vul-end>"):
                start = code.find(marker)
                if start >= 0:
                    spans.append((start, start + len(marker) - 1))
        spans.extend(required_spans)
        encoding = self.tokenizer(
            code,
            add_special_tokens=True,
            truncation=False,
            return_offsets_mapping=True,
        )
        input_ids = encoding["input_ids"]
        offsets = [tuple(offset) for offset in encoding["offset_mapping"]]
        content = [index for index, (start, end) in enumerate(offsets) if end > start]
        special = [index for index in range(len(offsets)) if index not in content]
        content_budget = max(0, self.node_max_length - len(special))
        selected_content = select_node_token_indices(
            offsets, content, spans, content_budget
        )
        selected = sorted((selected_content + special)[: self.node_max_length])
        selected_ids = [input_ids[index] for index in selected]
        selected_offsets = [offsets[index] for index in selected]
        padding = self.node_max_length - len(selected)
        return (
            torch.tensor(
                selected_ids + [self.tokenizer.pad_token_id] * padding,
                dtype=torch.long,
            ),
            torch.tensor([1] * len(selected) + [0] * padding, dtype=torch.long),
            selected_offsets + [(0, 0)] * padding,
        )

def select_node_token_indices(offsets, content, spans, budget):
    if budget <= 0:
        return []
    if len(content) <= budget:
        return content

    span_tokens = [
        [
            index
            for index in content
            if offsets[index][0] <= end and offsets[index][1] > start
        ]
        for start, end in spans
    ]
    anchors = list(
        dict.fromkeys(tokens[len(tokens) // 2] for tokens in span_tokens if tokens)
    )
    if len(anchors) > budget:
        return sorted(anchors[:budget])

    relevant = list(
        dict.fromkeys(
            index for tokens in span_tokens for index in tokens if index not in anchors
        )
    )
    selected = anchors + relevant[: budget - len(anchors)]
    remaining = budget - len(selected)
    if remaining:
        reference = selected or [content[0]]
        context = sorted(
            (index for index in content if index not in selected),
            key=lambda index: (
                min(abs(index - selected_index) for selected_index in reference),
                index,
            ),
        )
        selected.extend(context[:remaining])
    return sorted(selected)


def is_empty_vdg(vdg: VDG):
    return len(vdg.nodes) == 0


def get_vulnerable_node_ids(vdg: VDG) -> set[int]:
    ranges = vdg.graph.get("vulnerableRanges", [])
    if ranges:
        return {
            node.id
            for node in vdg.nodes
            if any(bounds["start"] <= node.id <= bounds["end"] for bounds in ranges)
        }
    return {
        node.id
        for node in vdg.nodes
        if "<vul-start>" in node.code or "<vul-end>" in node.code
    }


def _graph_distances(starts, adjacency):
    distances = {node_id: 0 for node_id in starts}
    queue = deque(starts)
    while queue:
        source = queue.popleft()
        for target in adjacency[source]:
            if target not in distances:
                distances[target] = distances[source] + 1
                queue.append(target)
    return distances


def _interleave(left, right):
    result = []
    for index in range(max(len(left), len(right))):
        if index < len(left):
            result.append(left[index])
        if index < len(right):
            result.append(right[index])
    return result


def select_vdg_nodes(vdg: VDG, max_nodes: Optional[int]) -> List[Node]:
    nodes = sorted(vdg.nodes, key=lambda node: node.id)
    if not max_nodes or len(nodes) <= max_nodes:
        return nodes

    vulnerable_ids = get_vulnerable_node_ids(vdg)
    if not vulnerable_ids:
        return nodes[:max_nodes]

    node_ids = {node.id for node in nodes}
    outgoing = defaultdict(set)
    incoming = defaultdict(set)
    edge_priority = defaultdict(int)
    for link in vdg.links:
        if (
            link.source != link.target
            and link.source in node_ids
            and link.target in node_ids
        ):
            outgoing[link.source].add(link.target)
            incoming[link.target].add(link.source)
            weight = 2 if link.type == "DDG" else 1
            edge_priority[link.source] += weight
            edge_priority[link.target] += weight

    downstream = _graph_distances(vulnerable_ids, outgoing)
    upstream = _graph_distances(vulnerable_ids, incoming)
    weak = _graph_distances(
        vulnerable_ids,
        {node_id: outgoing[node_id] | incoming[node_id] for node_id in node_ids},
    )
    context = {node.id: node for node in nodes if node.id not in vulnerable_ids}
    ordered = []
    line_distance = lambda node_id: min(
        abs(node_id - vulnerable_id) for vulnerable_id in vulnerable_ids
    )
    context_key = lambda node_id: (
        -edge_priority[node_id],
        line_distance(node_id),
        node_id,
    )
    for distance in sorted(set(weak.values()) - {0}):
        layer = [node_id for node_id in context if weak.get(node_id) == distance]
        both = sorted(
            (
                node_id
                for node_id in layer
                if node_id in upstream and node_id in downstream
            ),
            key=context_key,
        )
        up = sorted(
            (
                node_id
                for node_id in layer
                if node_id in upstream and node_id not in downstream
            ),
            key=context_key,
        )
        down = sorted(
            (
                node_id
                for node_id in layer
                if node_id in downstream and node_id not in upstream
            ),
            key=context_key,
        )
        other = sorted(
            (
                node_id
                for node_id in layer
                if node_id not in upstream and node_id not in downstream
            ),
            key=context_key,
        )
        ordered.extend(both + _interleave(up, down) + other)

    ordered.extend(
        sorted(
            (node_id for node_id in context if node_id not in weak),
            key=context_key,
        )
    )
    limit = max(max_nodes, len(vulnerable_ids))
    selected_ids = vulnerable_ids | set(ordered[: limit - len(vulnerable_ids)])
    return [node for node in nodes if node.id in selected_ids]


def prune_vdg(vdg: VDG, max_nodes: int) -> None:
    if max_nodes and len(vdg.nodes) > max_nodes:
        selected_nodes = select_vdg_nodes(vdg, max_nodes)
        selected_ids = {node.id for node in selected_nodes}
        vdg.nodes = selected_nodes
        vdg.links = [
            link
            for link in vdg.links
            if link.source in selected_ids and link.target in selected_ids
        ]


def _is_jsonl_path(p: Path) -> bool:
    # supports: .jsonl, .ndjson, .jsonl.gz, .ndjson.gz
    sfx = p.suffixes
    return (sfx[-1] in {".jsonl", ".ndjson"}) or (
        len(sfx) >= 2 and sfx[-2] in {".jsonl", ".ndjson"}
    )


def _is_compact_jsonl_path(p: Path) -> bool:
    return ".cjsonl" in p.suffixes


def _open_text(p: Path):
    return (
        gzip.open(p, "rt", encoding="utf-8")
        if p.suffix == ".gz"
        else open(p, "rt", encoding="utf-8")
    )


def load_data(
    data_path: str,
    max_nodes_per_graph: Optional[int] = None,
    drop_empty_graphs: bool = False,
) -> List[Sample]:
    path = Path(data_path)
    data = []

    with _open_text(path) as f:
        if _is_compact_jsonl_path(path):
            for sample in f:
                if sample:
                    data.append(decode_packed(sample))
        elif _is_jsonl_path(path):
            dec = msgspec.json.Decoder(type=Sample)
            for sample in f:
                if sample:
                    data.append(dec.decode(sample))
        else:
            data = msgspec.json.decode(f.read(), type=List[Sample])

    for k in range(len(data) - 1, -1, -1):
        example = data[k]
        example.id = k

        if drop_empty_graphs and is_empty_vdg(example.vdg):
            del data[k]
            continue

        if max_nodes_per_graph is not None:
            prune_vdg(example.vdg, max_nodes=max_nodes_per_graph)

    return data
