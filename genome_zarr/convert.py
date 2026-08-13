"""Streaming FASTA conversion and lossless packed Zarr transcoding."""

import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Tuple, Union

import numpy as np

from .codec import _BASE_CODES

SCHEMA_VERSION = 2
DEFAULT_CHUNK_BASES = 1_048_576
_MANIFEST_FILENAME = "manifest.json"
_PACKED_ARRAY_NAME = "packed_sequence"


@dataclass(frozen=True)
class _RecordSpan:
    name: str
    logical_length: int
    byte_offset: int
    byte_length: int


@dataclass
class _StoreIndex:
    layout: str
    group: object
    array_name: str | None
    array: object | None
    records: list[_RecordSpan]
    manifest: dict | None
    compressor: str


def _require_dependencies():
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - exercised by CLI users
        raise RuntimeError("Install dependencies with: pip install genome-zarr-4bit") from exc
    major_version = int(zarr.__version__.split(".", 1)[0])
    if major_version < 3:
        raise RuntimeError("Zarr v3 is required; install genome-zarr-4bit with zarr>=3")
    return zarr


def _prepare_destination(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {path} (use --overwrite to replace it)")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def _manifest_path(path: Path) -> Path:
    return path / _MANIFEST_FILENAME


def _write_manifest(path: Path, manifest: dict) -> None:
    with _manifest_path(path).open("wt", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_manifest(path: Path) -> dict | None:
    manifest_path = _manifest_path(path)
    if not manifest_path.is_file():
        return None
    with manifest_path.open("rt", encoding="utf-8") as handle:
        return json.load(handle)


def _record_name(header: str) -> str:
    name = header[1:].split()[0] if header[1:].split() else ""
    if not name:
        raise ValueError(f"FASTA header has an unsupported record name: {header!r}")
    # Zarr object keys cannot use path separators, so normalize the common
    # mmseqs-style coordinate suffix into a safe array name.
    name = name.replace("/", "_")
    if name in {".", ".."}:
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
                    # Add a duplicate suffix to avoid clobbering the first record.  This is a rare case, but it is possible to have a FASTA with two records that have the same name.  For example, if the FASTA was generated from a multi-FASTA file that had two records with the same name, or if the FASTA was generated from a reference genome that has multiple contigs with the same name.
                    current += "_duplicate_"+str(len([k for k in lengths.keys() if k.startswith(current)]))
                    print(f"Duplicate FASTA record name: {current} but duplicate suffix added to avoid clobbering the first record")
                lengths[current] = 0
            elif current is None:
                raise ValueError("FASTA sequence data appeared before the first header")
            else:
                lengths[current] += len(line)
    if not lengths:
        raise ValueError("FASTA contains no records")
    return lengths


def scan_fasta_indexed(fasta: Path) -> Dict[str, Tuple[int, int, int]]:
    """Scan FASTA in binary mode returning {record_name: (start_byte, end_byte, length_bases)}."""
    records = {}
    current_name = None
    start_byte = 0
    current_bases = 0

    with fasta.open("rb") as handle:
        while True:
            pos = handle.tell()
            line = handle.readline()
            if not line:
                if current_name is not None:
                    records[current_name] = (start_byte, pos, current_bases)
                break

            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(b">"):
                if current_name is not None:
                    records[current_name] = (start_byte, pos, current_bases)
                header = stripped.decode("ascii")
                current_name = _record_name(header)
                if current_name in records:
                    current_name += "_duplicate_"+str(len([k for k in records.keys() if k.startswith(current_name)]))
                    print(f"Duplicate FASTA record name: {current_name} but duplicate suffix added to avoid clobbering the first record")
                start_byte = handle.tell()
                current_bases = 0
            else:
                if current_name is None:
                    raise ValueError("FASTA sequence data appeared before the first header")
                current_bases += len(stripped)

    if not records:
        raise ValueError("FASTA contains no records")
    return records


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
    zarr = _require_dependencies()
    return zarr.codecs.ZstdCodec(level=level)


def _encode_record_bytes(fasta_path: Union[str, Path], start_byte: int, end_byte: int, logical_length: int) -> np.ndarray:
    encoded_bytes = (logical_length + 1) // 2
    data = np.empty(encoded_bytes, dtype=np.uint8)
    writer = _PackedArrayWriter(data, encoded_bytes)

    with Path(fasta_path).open("rb") as handle:
        handle.seek(start_byte)
        while handle.tell() < end_byte:
            line = handle.readline()
            if not line or line.startswith(b">"):
                break
            stripped = line.strip()
            if stripped:
                writer.feed(stripped.decode("ascii"))

    writer.finish()
    return data


class _PackedArrayWriter:
    """Pack arbitrary FASTA line boundaries and write to one packed array."""

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

    def finish_record(self) -> None:
        if self.pending is not None:
            self.buffer.append(self.pending << 4)
            self.pending = None

    def flush(self) -> None:
        if self.buffer:
            # ``frombuffer`` is a view of ``self.buffer``. Zarr can retain that
            # view for the duration of assignment, and resizing the bytearray
            # via ``clear`` would then raise BufferError.
            data = np.frombuffer(self.buffer, dtype=np.uint8).copy()
            self.array[self.position : self.position + data.size] = data
            self.position += data.size
            # Do not resize the old bytearray: a native codec may retain an
            # exported view of it briefly. Replacing it is always safe.
            self.buffer.clear()

    def finish(self) -> None:
        self.finish_record()
        self.flush()
        if self.position != self.array.shape[0]:
            raise RuntimeError("Packed output length did not match FASTA record length")


def _group_attributes(chunk_bases: int, compressor: str) -> dict:
    return {
        "schema": "genome-zarr-4bit",
        "schema_version": SCHEMA_VERSION,
        "architecture": "4bit",
        "storage_layout": "flat-packed",
        "array_name": _PACKED_ARRAY_NAME,
        "encoding": "two 4-bit base codes per uint8; first base is high nibble",
        "base_codes": {"A": 0, "C": 1, "G": 2, "T": 3, "N_or_unknown": 4},
        "logical_chunk_bases": chunk_bases,
        "compressor": compressor,
    }


def _manifest_from_records(records: list[tuple[str, int]], chunk_bases: int, compressor: str) -> dict:
    byte_offset = 0
    manifest_records = []
    total_bases = 0
    total_packed_bytes = 0
    for name, logical_length in records:
        byte_length = (logical_length + 1) // 2
        manifest_records.append(
            {
                "name": name,
                "logical_length": logical_length,
                "byte_offset": byte_offset,
                "byte_length": byte_length,
            }
        )
        byte_offset += byte_length
        total_bases += logical_length
        total_packed_bytes += byte_length
    return {
        "schema": "genome-zarr-4bit",
        "schema_version": SCHEMA_VERSION,
        "architecture": "4bit",
        "storage_layout": "flat-packed",
        "array_name": _PACKED_ARRAY_NAME,
        "compressor": compressor,
        "logical_chunk_bases": chunk_bases,
        "chunk_bytes": chunk_bases // 2,
        "encoding": "two 4-bit base codes per uint8; first base is high nibble",
        "base_codes": {"A": 0, "C": 1, "G": 2, "T": 3, "N_or_unknown": 4},
        "record_count": len(manifest_records),
        "logical_bases": total_bases,
        "packed_bytes": total_packed_bytes,
        "records": manifest_records,
    }


def _build_legacy_index(group) -> _StoreIndex:
    records = []
    for name in group.array_keys():
        array = group[name]
        if array.ndim != 1 or np.dtype(array.dtype) != np.dtype(np.uint8):
            raise ValueError(f"Array {name!r} is not a one-dimensional uint8 packed chromosome")
        logical_length = int(array.attrs.get("logical_length", int(array.shape[0]) * 2))
        records.append(_RecordSpan(name, logical_length, 0, int(array.shape[0])))
    if not records:
        raise ValueError("Source store contains no chromosome arrays")
    return _StoreIndex(
        layout="legacy-chromosome-arrays",
        group=group,
        array_name=None,
        array=None,
        records=records,
        manifest=None,
        compressor=str(group.attrs.get("compressor", "unknown")),
    )


def _load_store_index(store: Union[str, Path]) -> _StoreIndex:
    zarr = _require_dependencies()
    path = Path(store)
    group = zarr.open_group(str(path), mode="r")
    manifest = _read_manifest(path)
    if manifest is not None:
        if manifest.get("schema") != "genome-zarr-4bit" or manifest.get("architecture") != "4bit":
            raise ValueError("Source metadata is not a genome-zarr-4bit manifest")
        if manifest.get("storage_layout") != "flat-packed":
            raise ValueError("Source metadata does not describe the flat-packed layout")
        array_name = manifest.get("array_name", _PACKED_ARRAY_NAME)
        try:
            array = group[array_name]
        except KeyError as exc:
            raise ValueError(f"Source store is missing packed array: {array_name}") from exc
        records = [
            _RecordSpan(
                name=record["name"],
                logical_length=int(record["logical_length"]),
                byte_offset=int(record["byte_offset"]),
                byte_length=int(record["byte_length"]),
            )
            for record in manifest.get("records", [])
        ]
        if not records:
            raise ValueError("Source store contains no chromosome records")
        return _StoreIndex(
            layout="flat-packed",
            group=group,
            array_name=array_name,
            array=array,
            records=records,
            manifest=manifest,
            compressor=str(manifest.get("compressor", "unknown")),
        )
    if group.attrs.get("schema") == "genome-zarr-4bit" and group.attrs.get("architecture") == "4bit":
        return _build_legacy_index(group)
    raise ValueError("Source is not a genome-zarr-4bit store created by this package")


def store_statistics(store: Union[str, Path]) -> dict:
    """Return logical and filesystem-size statistics for a packed store."""
    index = _load_store_index(store)
    path = Path(store)

    if index.layout == "flat-packed":
        packed_bytes = int(index.array.shape[0])
        logical_bases = int(index.manifest["logical_bases"])
        chromosomes = len(index.records)
        zarr_format = index.group.metadata.zarr_format
        compressor = index.manifest.get("compressor", "unknown")
    else:
        chromosomes = 0
        logical_bases = 0
        packed_bytes = 0
        for record in index.records:
            array = index.group[record.name]
            chromosomes += 1
            packed_bytes += int(array.shape[0])
            logical_bases += int(array.attrs.get("logical_length", array.shape[0] * 2))
        zarr_format = index.group.metadata.zarr_format
        compressor = index.group.attrs.get("compressor", "unknown")

    apparent_bytes = 0
    allocated_bytes = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            stat = entry.stat()
            apparent_bytes += stat.st_size
            allocated_bytes += getattr(stat, "st_blocks", 0) * 512
    return {
        "chromosomes": chromosomes,
        "logical_bases": logical_bases,
        "packed_bytes": packed_bytes,
        "apparent_bytes": apparent_bytes,
        "allocated_bytes": allocated_bytes,
        "compressor": compressor,
        "zarr_format": zarr_format,
    }


_WORKER_GROUP = None
_WORKER_ARRAY = None


def _init_chunk_worker(output_path: str, array_name: str) -> None:
    global _WORKER_GROUP, _WORKER_ARRAY
    import zarr

    _WORKER_GROUP = zarr.open_group(str(output_path), mode="a")
    _WORKER_ARRAY = _WORKER_GROUP[array_name]


def _encode_fasta_pieces_from_handle(handle, pieces: list[tuple[int, int]], logical_length: int) -> np.ndarray:
    """Encode one byte-aligned part of a FASTA record.

    ``pieces`` are physical spans containing only sequence characters.  A
    part is never shared by two output chunks, which lets workers write Zarr
    chunks independently.
    """
    sequence = bytearray()
    for start_byte, end_byte in pieces:
        handle.seek(start_byte)
        sequence.extend(handle.read(end_byte - start_byte))
    if len(sequence) != logical_length:
        raise RuntimeError("Indexed FASTA span did not match its expected length")
    codes = _BASE_CODES[np.frombuffer(sequence, dtype=np.uint8)]
    if codes.size % 2:
        codes = np.pad(codes, (0, 1), constant_values=0)
    return (codes[0::2] << 4) | codes[1::2]


def _encode_fasta_pieces(fasta_path: str, pieces: list[tuple[int, int]], logical_length: int) -> np.ndarray:
    with Path(fasta_path).open("rb") as handle:
        return _encode_fasta_pieces_from_handle(handle, pieces, logical_length)


def _write_chunk_worker(args: tuple) -> tuple:
    fasta_path, chunk_index, expected_bytes, record_parts = args
    if _WORKER_ARRAY is None:
        raise RuntimeError("Worker array was not initialized")

    with Path(fasta_path).open("rb") as handle:
        encoded_parts = [
            _encode_fasta_pieces_from_handle(handle, pieces, length)
            for pieces, length in record_parts
        ]
    data = np.concatenate(encoded_parts) if encoded_parts else np.empty(0, dtype=np.uint8)
    if data.size != expected_bytes:
        raise RuntimeError("Packed output chunk length did not match its expected length")
    # A worker owns the complete Zarr chunk.  Slice assignment would perform
    # read-modify-write when a record crosses a chunk boundary, which both
    # serializes workers and risks clobbering a concurrent write.
    _WORKER_ARRAY.blocks[chunk_index] = data
    return chunk_index, data.size


def _build_fasta_chunk_tasks(
    fasta: Path, records: Dict[str, Tuple[int, int, int]], chunk_bytes: int
) -> list[tuple[int, int, list[tuple[list[tuple[int, int]], int]]]]:
    """Map FASTA sequence spans to independent packed-output chunks.

    Records are padded independently when their length is odd.  Keeping a
    separate part per record preserves that convention even when a Zarr chunk
    contains the end of one record and the beginning of the next.
    """
    tasks = []
    chunk_index = 0
    chunk_used = 0
    chunk_parts: list[tuple[list[tuple[int, int]], int]] = []

    def finish_chunk() -> None:
        nonlocal chunk_index, chunk_used, chunk_parts
        if chunk_used:
            tasks.append((chunk_index, chunk_used, chunk_parts))
            chunk_index += 1
            chunk_used = 0
            chunk_parts = []

    with fasta.open("rb") as handle:
        for name, (record_start, record_end, record_length) in records.items():
            handle.seek(record_start)
            remaining_record = record_length
            part_pieces: list[tuple[int, int]] = []
            part_bases = 0
            while handle.tell() < record_end:
                line_start = handle.tell()
                line = handle.readline()
                if not line or line.startswith(b">"):
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                # scan_fasta_indexed uses strip() too.  The stored span omits
                # only leading/trailing whitespace and retains internal bytes.
                content_start = line_start + (len(line) - len(line.lstrip()))
                cursor = 0
                while cursor < len(stripped):
                    capacity = 2 * (chunk_bytes - chunk_used) - part_bases
                    if capacity <= 0:
                        raise RuntimeError("FASTA chunk planner lost byte alignment")
                    take = min(len(stripped) - cursor, remaining_record, capacity)
                    part_pieces.append((content_start + cursor, content_start + cursor + take))
                    cursor += take
                    remaining_record -= take
                    part_bases += take
                    if part_bases == 2 * (chunk_bytes - chunk_used):
                        # A chunk boundary is always byte aligned, so this
                        # non-final piece has an even number of bases.
                        chunk_parts.append((part_pieces, part_bases))
                        chunk_used = chunk_bytes
                        part_pieces = []
                        part_bases = 0
                        finish_chunk()
            if remaining_record:
                raise RuntimeError(f"Indexed FASTA record {name!r} ended early")
            if part_bases:
                chunk_parts.append((part_pieces, part_bases))
                chunk_used += (part_bases + 1) // 2
                if chunk_used == chunk_bytes:
                    finish_chunk()

    finish_chunk()
    return tasks


def fasta_to_zstd(
    fasta_path: Union[str, Path],
    destination: Union[str, Path],
    *,
    chunk_bases: int = DEFAULT_CHUNK_BASES,
    zstd_level: int = 3,
    overwrite: bool = False,
    num_workers: int = None,
) -> Path:
    """Create a Zstandard-compressed flat-packed Zarr store from FASTA."""
    if chunk_bases <= 0 or chunk_bases % 2:
        raise ValueError("chunk_bases must be a positive even number")
    if num_workers is not None and num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    fasta, output = Path(fasta_path), Path(destination)
    if not fasta.is_file():
        raise FileNotFoundError(f"FASTA file does not exist: {fasta}")

    records = scan_fasta_indexed(fasta)
    ordered_records = [(name, length) for name, (_, _, length) in records.items()]
    total_bases = sum(length for _, length in ordered_records)
    total_packed_bytes = sum((length + 1) // 2 for _, length in ordered_records)

    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 16)

    print(f"Converting {len(ordered_records)} FASTA records ({total_bases} bases, {total_packed_bytes} packed bytes) to Zarr store: {output} with {num_workers} workers", flush=True)
    zarr = _require_dependencies()
    _prepare_destination(output, overwrite)
    group = zarr.open_group(str(output), mode="w", zarr_format=3)
    group.attrs.update(_group_attributes(chunk_bases, "zstd"))

    chunk_bytes = chunk_bases // 2
    array = group.create_array(
        _PACKED_ARRAY_NAME,
        shape=(total_packed_bytes,),
        chunks=(chunk_bytes,),
        dtype=np.uint8,
        compressors=[_zstd(zstd_level)],
    )
    array.attrs.update({
        "logical_length": total_bases,
        "encoding": "4bit",
        "storage_layout": "flat-packed",
    })

    print(f"Writing packed sequence array with {array.nchunks} chunks of {chunk_bytes} bytes each", flush=True)

    if num_workers <= 1:
        offset = 0
        for name, (start_byte, end_byte, length) in records.items():
            data = _encode_record_bytes(fasta, start_byte, end_byte, length)
            array[offset : offset + data.size] = data
            offset += data.size
    else:
        chunk_tasks = _build_fasta_chunk_tasks(fasta, records, chunk_bytes)
        worker_tasks = [(str(fasta), *task) for task in chunk_tasks]
        total_chunks = len(worker_tasks)
        completed = 0
        if total_chunks:
            with ProcessPoolExecutor(
                max_workers=min(num_workers, total_chunks),
                initializer=_init_chunk_worker,
                initargs=(str(output), _PACKED_ARRAY_NAME),
            ) as executor:
                futures = [executor.submit(_write_chunk_worker, task) for task in worker_tasks]
                for future in as_completed(futures):
                    future.result()
                    completed += 1
                    pct = (completed / total_chunks) * 100
                    print(f"Progress: {completed}/{total_chunks} chunks converted ({pct:.1f}%)", flush=True)

    manifest = _manifest_from_records(ordered_records, chunk_bases, "zstd")
    _write_manifest(output, manifest)
    return output


def _validate_packed_group(group) -> None:
    if getattr(group.metadata, "zarr_format", None) not in (2, 3):
        raise ValueError("Source is not a supported Zarr v2 or v3 store")
    if group.attrs.get("schema") != "genome-zarr-4bit" or group.attrs.get("architecture") != "4bit":
        raise ValueError("Source is not a genome-zarr-4bit store created by this package")
    if group.attrs.get("storage_layout") == "flat-packed":
        if _PACKED_ARRAY_NAME not in list(group.array_keys()):
            raise ValueError("Source store contains no packed sequence array")
    elif not list(group.array_keys()):
        raise ValueError("Source store contains no chromosome arrays")


def _copy_slice_worker(args: tuple) -> None:
    source_path, destination_path, source_array_name, destination_array_name, start, end, destination_start = args

    import zarr

    source_group = zarr.open_group(str(source_path), mode="r")
    destination_group = zarr.open_group(str(destination_path), mode="a")
    source_array = source_group[source_array_name]
    destination_array = destination_group[destination_array_name]
    destination_array[destination_start:destination_start + (end - start)] = source_array[start:end]


def _transcode(
    source: Union[str, Path], destination: Union[str, Path], compressor, compressor_name: str, overwrite: bool, num_workers: int = None
) -> Path:
    zarr = _require_dependencies()
    source_path, output = Path(source), Path(destination)
    if not source_path.is_dir():
        raise FileNotFoundError(f"Zarr source does not exist: {source_path}")
    if source_path.resolve() == output.resolve():
        raise ValueError("Source and destination must be different stores")

    source_index = _load_store_index(source_path)
    _prepare_destination(output, overwrite)
    destination_group = zarr.open_group(str(output), mode="w", zarr_format=3)

    if source_index.layout == "flat-packed":
        source_manifest = dict(source_index.manifest or {})
        output_chunk_bytes = int(source_index.array.chunks[0] if source_index.array.chunks else source_index.array.shape[0])
        total_packed_bytes = int(source_index.array.shape[0])
        destination_array = destination_group.create_array(
            _PACKED_ARRAY_NAME,
            shape=(total_packed_bytes,),
            chunks=(output_chunk_bytes,),
            dtype=np.uint8,
            compressors=compressor,
        )
        destination_array.attrs.update(dict(source_index.array.attrs))
        destination_group.attrs.update(dict(source_index.group.attrs))
        destination_group.attrs["compressor"] = compressor_name
        destination_group.attrs["storage_layout"] = "flat-packed"
        destination_group.attrs["array_name"] = _PACKED_ARRAY_NAME
        source_manifest["compressor"] = compressor_name
        source_manifest["chunk_bytes"] = output_chunk_bytes
        source_manifest["logical_chunk_bases"] = output_chunk_bytes * 2
        worker_tasks = []
        step = output_chunk_bytes
        for start in range(0, total_packed_bytes, step):
            end = min(start + step, total_packed_bytes)
            worker_tasks.append((str(source_path), str(output), _PACKED_ARRAY_NAME, _PACKED_ARRAY_NAME, start, end, start))
        copy_records = source_index.records
    else:
        total_packed_bytes = sum(record.byte_length for record in source_index.records)
        first_record = source_index.records[0]
        first_array = source_index.group[first_record.name]
        output_chunk_bytes = int(first_array.chunks[0] if first_array.chunks else first_array.shape[0])
        if output_chunk_bytes <= 0:
            output_chunk_bytes = DEFAULT_CHUNK_BASES // 2
        destination_array = destination_group.create_array(
            _PACKED_ARRAY_NAME,
            shape=(total_packed_bytes,),
            chunks=(output_chunk_bytes,),
            dtype=np.uint8,
            compressors=compressor,
        )
        destination_group.attrs.update(_group_attributes(output_chunk_bytes * 2, compressor_name))
        worker_tasks = []
        destination_offset = 0
        for record in source_index.records:
            source_array = source_index.group[record.name]
            step = int(source_array.chunks[0] if source_array.chunks else source_array.shape[0])
            for start in range(0, int(source_array.shape[0]), step):
                end = min(start + step, int(source_array.shape[0]))
                worker_tasks.append((str(source_path), str(output), record.name, _PACKED_ARRAY_NAME, start, end, destination_offset + start))
            destination_offset += int(source_array.shape[0])
        copy_records = [
            _RecordSpan(record.name, record.logical_length, record.byte_offset, record.byte_length)
            for record in source_index.records
        ]

    if source_index.layout == "flat-packed":
        destination_group.attrs.update(dict(source_index.group.attrs))
        destination_group.attrs["compressor"] = compressor_name
        destination_group.attrs["storage_layout"] = "flat-packed"
        destination_group.attrs["array_name"] = _PACKED_ARRAY_NAME
        destination_group.attrs["logical_chunk_bases"] = output_chunk_bytes * 2
        destination_group.attrs["schema_version"] = SCHEMA_VERSION
        destination_group.attrs["schema"] = "genome-zarr-4bit"
        destination_group.attrs["architecture"] = "4bit"
        destination_group.attrs["encoding"] = "two 4-bit base codes per uint8; first base is high nibble"
        destination_group.attrs["base_codes"] = {"A": 0, "C": 1, "G": 2, "T": 3, "N_or_unknown": 4}
        destination_group[_PACKED_ARRAY_NAME].attrs.update(dict(source_index.array.attrs))
        destination_group[_PACKED_ARRAY_NAME].attrs["storage_layout"] = "flat-packed"
        destination_group[_PACKED_ARRAY_NAME].attrs["logical_length"] = int(
            source_index.manifest.get("logical_bases", destination_group[_PACKED_ARRAY_NAME].shape[0] * 2)
        )

    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 16)

    print(f"Transcoding {len(copy_records)} records ({total_packed_bytes} packed bytes) from {source_path} to {output} with {num_workers} workers", flush=True)
    total = len(worker_tasks)
    completed = 0
    if total:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_copy_slice_worker, task) for task in worker_tasks]
            for future in as_completed(futures):
                future.result()
                completed += 1
                pct = (completed / total) * 100
                print(f"Progress: {completed}/{total} slices copied ({pct:.1f}%)", flush=True)

    manifest = _manifest_from_records(
        [(record.name, record.logical_length) for record in copy_records],
        output_chunk_bytes * 2,
        compressor_name,
    )
    _write_manifest(output, manifest)
    return output


def decompress_zarr(
    source: Union[str, Path], destination: Union[str, Path], *, overwrite: bool = False, num_workers: int = None
) -> Path:
    """Copy a compressed packed store into an otherwise identical uncompressed store."""
    return _transcode(source, destination, None, "none", overwrite, num_workers=num_workers)


def compress_zarr(
    source: Union[str, Path], destination: Union[str, Path], *, zstd_level: int = 3, overwrite: bool = False, num_workers: int = None
) -> Path:
    """Copy an uncompressed packed store (or recompress any packed store) with Zstd."""
    return _transcode(source, destination, _zstd(zstd_level), "zstd", overwrite, num_workers=num_workers)
