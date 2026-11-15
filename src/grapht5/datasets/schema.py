from typing import Any, Dict, List, Optional

import msgspec


class Characters(msgspec.Struct):
    start: int
    end: int


class Link(msgspec.Struct):
    source: int
    label: str
    dstCharacters: Characters
    key: str
    target: int
    type: str
    srcCharacters: Characters


class Node(msgspec.Struct):
    id: int
    code: str


class VDG(msgspec.Struct):
    directed: bool
    links: List[Link]
    nodes: List[Node]
    graph: Dict[str, Any]
    multigraph: bool


class Passage(msgspec.Struct, kw_only=True):
    id: int
    type: Optional[str] = None
    prefix: str
    text: str


class Sample(msgspec.Struct, kw_only=True):
    id: int
    vul4j: Optional[int] = None
    cwe_id: Optional[str] = None
    cve_id: Optional[str] = None
    repair_group_id: Optional[str] = None
    function_repair_id: Optional[str] = None
    function_name: Optional[str] = None
    source_file_path: Optional[str] = None
    multi_file_patch: Optional[bool] = None
    num_changed_files: Optional[int] = None
    changed_files: Optional[List[str]] = None
    original_address: Optional[str] = None
    passages: List[Passage]
    target: str
    vdg: VDG


class PackedCharacters(msgspec.Struct, array_like=True):
    start: int
    end: int


class PackedLink(msgspec.Struct, array_like=True):
    source: int
    target: int
    type: str
    label: str
    src_characters: PackedCharacters
    dst_characters: PackedCharacters


class PackedNode(msgspec.Struct, array_like=True):
    id: int
    code: str


class PackedVDG(msgspec.Struct, array_like=True):
    directed: bool
    links: List[PackedLink]
    nodes: List[PackedNode]
    graph: Dict[str, Any]
    multigraph: bool


class PackedPassage(msgspec.Struct, array_like=True):
    id: int
    type: Optional[str]
    prefix: str
    text: str


class PackedSample(msgspec.Struct, array_like=True):
    id: int
    passages: List[PackedPassage]
    target: str
    vdg: PackedVDG
    vul4j: Optional[int] = None
    cwe_id: Optional[str] = None
    cve_id: Optional[str] = None
    repair_group_id: Optional[str] = None
    function_repair_id: Optional[str] = None
    function_name: Optional[str] = None
    source_file_path: Optional[str] = None
    multi_file_patch: Optional[bool] = None
    num_changed_files: Optional[int] = None
    changed_files: Optional[List[str]] = None
    original_address: Optional[str] = None


def pack_sample(sample: Sample) -> PackedSample:
    return PackedSample(
        sample.id,
        [PackedPassage(p.id, p.type, p.prefix, p.text) for p in sample.passages],
        sample.target,
        PackedVDG(
            sample.vdg.directed,
            [
                PackedLink(
                    link.source,
                    link.target,
                    link.type,
                    link.label,
                    PackedCharacters(link.srcCharacters.start, link.srcCharacters.end),
                    PackedCharacters(link.dstCharacters.start, link.dstCharacters.end),
                )
                for link in sample.vdg.links
            ],
            [PackedNode(node.id, node.code) for node in sample.vdg.nodes],
            sample.vdg.graph,
            sample.vdg.multigraph,
        ),
        sample.vul4j,
        sample.cwe_id,
        sample.cve_id,
        sample.repair_group_id,
        sample.function_repair_id,
        sample.function_name,
        sample.source_file_path,
        sample.multi_file_patch,
        sample.num_changed_files,
        sample.changed_files,
        sample.original_address,
    )


def unpack_sample(sample: PackedSample) -> Sample:
    return Sample(
        id=sample.id,
        passages=[
            Passage(id=p.id, type=p.type, prefix=p.prefix, text=p.text)
            for p in sample.passages
        ],
        target=sample.target,
        vdg=VDG(
            sample.vdg.directed,
            [
                Link(
                    link.source,
                    link.label,
                    Characters(link.dst_characters.start, link.dst_characters.end),
                    f"{link.source}-{link.target}-{link.type}-{link.label}",
                    link.target,
                    link.type,
                    Characters(link.src_characters.start, link.src_characters.end),
                )
                for link in sample.vdg.links
            ],
            [Node(node.id, node.code) for node in sample.vdg.nodes],
            sample.vdg.graph,
            sample.vdg.multigraph,
        ),
        vul4j=sample.vul4j,
        cwe_id=sample.cwe_id,
        cve_id=sample.cve_id,
        repair_group_id=sample.repair_group_id,
        function_repair_id=sample.function_repair_id,
        function_name=sample.function_name,
        source_file_path=sample.source_file_path,
        multi_file_patch=sample.multi_file_patch,
        num_changed_files=sample.num_changed_files,
        changed_files=sample.changed_files,
        original_address=sample.original_address,
    )
