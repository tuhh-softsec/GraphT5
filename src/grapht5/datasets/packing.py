import gzip
import os
from dataclasses import dataclass
from pathlib import Path

import msgspec

from grapht5.datasets.schema import PackedSample, Sample, pack_sample, unpack_sample

_packed_decoder = msgspec.json.Decoder(type=PackedSample)


@dataclass(frozen=True)
class PackingStats:
    samples: int
    input_bytes: int
    output_bytes: int


def _open_binary(path: Path, mode: str):
    return gzip.open(path, mode) if path.suffix == ".gz" else open(path, mode)


def pack_jsonl(source: Path, destination: Path) -> PackingStats:
    decoder = msgspec.json.Decoder(type=Sample)
    encoder = msgspec.json.Encoder()
    compressed_suffix = ".gz" if destination.suffix == ".gz" else ""
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp{compressed_suffix}"
    )
    samples = 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        _open_binary(source, "rb") as input_file,
        _open_binary(temporary, "wb") as output_file,
    ):
        for line in input_file:
            if line.strip():
                output_file.write(encoder.encode(pack_sample(decoder.decode(line))))
                output_file.write(b"\n")
                samples += 1

    temporary.replace(destination)
    return PackingStats(samples, source.stat().st_size, destination.stat().st_size)


def decode_packed(line: bytes | str) -> Sample:
    return unpack_sample(_packed_decoder.decode(line))
