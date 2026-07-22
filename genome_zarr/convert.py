"""Streaming FASTA conversion and lossless Zarr transcoding."""

import shutil
from pathlib import Path
from typing import Dict, Iterator, Tuple, Union

import numpy as np

from .codec import _BASE_CODES

SCHEMA_VERSION = 1
DEFAULT_CHUNK_BASES = 1_048_576


def _require_dependencies():
    try:
        import numcodecs
        import zarr
    except ImportError as exc:  # pragma: no cover - exercised by CLI users
        raise RuntimeError("Install dependencies with: pip install genome-zarr-4bit") from exc
    return zarr, numcodecs


def _prepare_destination(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {path} (use --overwrite to replace it)")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def _record_name(header: str) -> str:
    name = header[1:].split()[0] if header[1:].split() else ""
    if not name or name in {".", ".."} or "/" in name:
        raise ValueError(f"FASTA header has an unsupported record name: {header!r}")
    return name


def scan_fasta(fasta: Path) -> Dict[str, int]:
    """Return record lengths without holding sequence data in memory."""
    lengths: Dict[str, int] = {}
    current = None
    with fasta.open("rt", encoding="ascii") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = _record_name(line)
                if current in lengths:
                    raise ValueError(f"Duplicate FASTA record name: {current}")
                lengths[current] = 0
            elif current is None:
                raise ValueError("FASTA sequence data appeared before the first header")
            else:
                lengths[current] += len(line)
    if not lengths:
        raise ValueError("FASTA contains no records")
    return lengths


def iter_fasta_lines(fasta: Path) -> Iterator[Tuple[str, str]]:
    """Yield (record name, sequence line), preserving streaming behaviour."""
    current = None
    with fasta.open("rt", encoding="ascii") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = _record_name(line)
            elif current is None:
                raise ValueError("FASTA sequence data appeared before the first header")
            else:
                yield current, line


def _zstd(level: int):
    _, numcodecs = _require_dependencies()
    return numcodecs.Zstd(level=level)


class _PackedArrayWriter:
    """Pack arbitrary FASTA line boundaries and write bounded Zarr slices."""

    def __init__(self, array, flush_bytes: int):
        self.array = array
        self.flush_bytes = flush_bytes
        self.pending = None
        self.buffer = bytearray()
        self.position = 0

    def feed(self, sequence: str) -> None:
        codes = _BASE_CODES[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
        if self.pending is not None:
            if codes.size:
                self.buffer.append((self.pending << 4) | int(codes[0]))
                self.pending = None
                codes = codes[1:]
        pair_count = codes.size // 2
        if pair_count:
            pairs = (codes[: pair_count * 2 : 2] << 4) | codes[1 : pair_count * 2 : 2]
            self.buffer.extend(pairs.tobytes())
        if codes.size % 2:
            self.pending = int(codes[-1])
        if len(self.buffer) >= self.flush_bytes:
            self.flush()

    def flush(self) -> None:
        if self.buffer:
            # ``frombuffer`` is a view of ``self.buffer``.  Zarr can retain
            # that view for the duration of assignment, and resizing the
            # bytearray via ``clear`` would then raise BufferError.  Make the
            # chunk owned before releasing the reusable write buffer.
            data = np.frombuffer(self.buffer, dtype=np.uint8).copy()
            self.array[self.position : self.position + data.size] = data
            self.position += data.size
            self.buffer.clear()

    def finish(self) -> None:
        if self.pending is not None:
            self.buffer.append(self.pending << 4)
            self.pending = None
        self.flush()
        if self.position != self.array.shape[0]:
            raise RuntimeError("Packed output length did not match FASTA record length")


def _group_attributes(chunk_bases: int, compressor: str) -> dict:
    return {
        "schema": "genome-zarr-4bit",
        "schema_version": SCHEMA_VERSION,
        "architecture": "4bit",
        "encoding": "two 4-bit base codes per uint8; first base is high nibble",
        "base_codes": {"A": 0, "C": 1, "G": 2, "T": 3, "N_or_unknown": 4},
        "logical_chunk_bases": chunk_bases,
        "compressor": compressor,
    }


def fasta_to_zstd(
    fasta_path: Union[str, Path],
    destination: Union[str, Path],
    *,
    chunk_bases: int = DEFAULT_CHUNK_BASES,
    zstd_level: int = 3,
    overwrite: bool = False,
) -> Path:
    """Create a Zstandard-compressed packed-4-bit Zarr group from FASTA."""
    if chunk_bases <= 0 or chunk_bases % 2:
        raise ValueError("chunk_bases must be a positive even number")
    fasta, output = Path(fasta_path), Path(destination)
    if not fasta.is_file():
        raise FileNotFoundError(f"FASTA file does not exist: {fasta}")
    lengths = scan_fasta(fasta)
    zarr, _ = _require_dependencies()
    _prepare_destination(output, overwrite)
    group = zarr.open_group(str(output), mode="w")
    group.attrs.update(_group_attributes(chunk_bases, "zstd"))

    arrays = {}
    chunk_bytes = chunk_bases // 2
    for name, length in lengths.items():
        encoded_bytes = (length + 1) // 2
        array = group.create_dataset(
            name,
            shape=(encoded_bytes,),
            chunks=(min(chunk_bytes, max(encoded_bytes, 1)),),
            dtype=np.uint8,
            compressor=_zstd(zstd_level),
        )
        array.attrs.update({"logical_length": length, "encoding": "4bit"})
        arrays[name] = _PackedArrayWriter(array, chunk_bytes)

    for name, sequence_line in iter_fasta_lines(fasta):
        arrays[name].feed(sequence_line)
    for writer in arrays.values():
        writer.finish()
    return output


def _validate_packed_group(group) -> None:
    if group.attrs.get("schema") != "genome-zarr-4bit" or group.attrs.get("architecture") != "4bit":
        raise ValueError("Source is not a genome-zarr-4bit store created by this package")
    if not list(group.array_keys()):
        raise ValueError("Source store contains no chromosome arrays")


def _transcode(
    source: Union[str, Path], destination: Union[str, Path], compressor, compressor_name: str, overwrite: bool
) -> Path:
    zarr, _ = _require_dependencies()
    source_path, output = Path(source), Path(destination)
    if not source_path.is_dir():
        raise FileNotFoundError(f"Zarr source does not exist: {source_path}")
    if source_path.resolve() == output.resolve():
        raise ValueError("Source and destination must be different stores")
    source_group = zarr.open_group(str(source_path), mode="r")
    _validate_packed_group(source_group)
    _prepare_destination(output, overwrite)
    destination_group = zarr.open_group(str(output), mode="w")
    destination_group.attrs.update(dict(source_group.attrs))
    destination_group.attrs["compressor"] = compressor_name

    for name in source_group.array_keys():
        source_array = source_group[name]
        if source_array.ndim != 1 or source_array.dtype != np.dtype("uint8"):
            raise ValueError(f"Array {name!r} is not a one-dimensional uint8 packed chromosome")
        target = destination_group.create_dataset(
            name,
            shape=source_array.shape,
            chunks=source_array.chunks,
            dtype=np.uint8,
            compressor=compressor,
        )
        target.attrs.update(dict(source_array.attrs))
        step = source_array.chunks[0]
        for start in range(0, source_array.shape[0], step):
            end = min(start + step, source_array.shape[0])
            target[start:end] = source_array[start:end]
    return output


def decompress_zarr(source: Union[str, Path], destination: Union[str, Path], *, overwrite: bool = False) -> Path:
    """Copy a compressed packed store into an otherwise identical uncompressed store."""
    return _transcode(source, destination, None, "none", overwrite)


def compress_zarr(
    source: Union[str, Path], destination: Union[str, Path], *, zstd_level: int = 3, overwrite: bool = False
) -> Path:
    """Copy an uncompressed packed store (or recompress any packed store) with Zstd."""
    return _transcode(source, destination, _zstd(zstd_level), "zstd", overwrite)
