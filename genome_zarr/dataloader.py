"""PyTorch-compatible window loading for ``genome-zarr-4bit`` stores.

The dataset intentionally opens Zarr lazily in each worker. Zarr array and
store objects should not be inherited by forked DataLoader workers, and this
also keeps the dataset pickle-safe when the ``spawn`` start method is used.
"""

from __future__ import annotations

import json
import random
from bisect import bisect_right
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Sequence

import numpy as np

_MANIFEST_FILENAME = "manifest.json"
_PACKED_ARRAY_NAME = "packed_sequence"

try:  # Keep the package usable for conversion-only installations.
    from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
except ImportError:  # pragma: no cover - depends on the caller's environment
    DataLoader = None
    get_worker_info = None

    class Dataset:  # type: ignore[no-redef]
        """Fallback base class when PyTorch is not installed."""

    class IterableDataset:  # type: ignore[no-redef]
        """Fallback base class when PyTorch is not installed."""


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
    compressor: str


def _manifest_path(store_path: Path) -> Path:
    return store_path / _MANIFEST_FILENAME


def _read_manifest(store_path: Path) -> dict | None:
    path = _manifest_path(store_path)
    if not path.is_file():
        return None
    with path.open("rt", encoding="utf-8") as handle:
        return json.load(handle)


def _open_group(store_path: str | Path):
    import zarr

    return zarr.open_group(str(Path(store_path)), mode="r")


def _decode_packed(packed: np.ndarray) -> np.ndarray:
    bases = np.empty(packed.size * 2, dtype=np.uint8)
    bases[0::2] = packed >> 4
    bases[1::2] = packed & 0x0F
    return bases


def _load_store_index(store_path: str | Path) -> _StoreIndex:
    path = Path(store_path)
    group = _open_group(path)
    manifest = _read_manifest(path)

    if manifest is not None:
        if manifest.get("schema") != "genome-zarr-4bit" or manifest.get("storage_layout") != "flat-packed":
            raise ValueError("Store manifest is not a flat-packed genome-zarr-4bit store")
        array_name = manifest.get("array_name", _PACKED_ARRAY_NAME)
        try:
            array = group[array_name]
        except KeyError as exc:
            raise ValueError(f"Store is missing packed array: {array_name}") from exc
        if array.ndim != 1 or np.dtype(array.dtype) != np.dtype(np.uint8):
            raise ValueError(f"Array {array_name!r} is not a one-dimensional uint8 packed sequence")
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
            raise ValueError("Store contains no chromosome records")
        return _StoreIndex(
            layout="flat-packed",
            group=group,
            array_name=array_name,
            array=array,
            records=records,
            compressor=str(manifest.get("compressor", "unknown")),
        )

    if group.attrs.get("schema") != "genome-zarr-4bit" or group.attrs.get("architecture") != "4bit":
        raise ValueError("Store is not a genome-zarr-4bit store created by genome-zarr")

    records = []
    for name in group.array_keys():
        array = group[name]
        if array.ndim != 1 or np.dtype(array.dtype) != np.dtype(np.uint8):
            raise ValueError(f"Array {name!r} is not a one-dimensional uint8 packed chromosome")
        logical_length = int(array.attrs.get("logical_length", int(array.shape[0]) * 2))
        records.append(_RecordSpan(name, logical_length, 0, int(array.shape[0])))
    if not records:
        raise ValueError("Store contains no chromosome arrays")
    return _StoreIndex(
        layout="legacy-chromosome-arrays",
        group=group,
        array_name=None,
        array=None,
        records=records,
        compressor=str(group.attrs.get("compressor", "unknown")),
    )


class GenomeZarrDataset(Dataset):
    """Load fixed-length base-code windows from a packed genome store."""

    def __init__(
        self,
        store_path: str | Path,
        window_size: int,
        *,
        stride: int | None = None,
        chromosomes: str | Sequence[str] | None = None,
        return_metadata: bool = False,
    ) -> None:
        if not isinstance(window_size, Integral) or isinstance(window_size, bool) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if stride is None:
            stride = window_size
        if not isinstance(stride, Integral) or isinstance(stride, bool) or stride <= 0:
            raise ValueError("stride must be a positive integer")

        self.store_path = str(Path(store_path))
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.return_metadata = return_metadata
        self._group = None
        self._arrays = {}

        index = _load_store_index(self.store_path)
        self._layout = index.layout
        self._array_name = index.array_name
        self.compressor = index.compressor
        self._records_by_name = {record.name: record for record in index.records}

        if chromosomes is None:
            selected = sorted(self._records_by_name)
        elif isinstance(chromosomes, str):
            selected = [chromosomes]
        else:
            selected = list(chromosomes)
        if not selected:
            raise ValueError("No chromosomes were selected")

        records: list[tuple[str, int, int, int, int]] = []
        for name in selected:
            record = self._records_by_name.get(name)
            if record is None:
                raise KeyError(f"Chromosome not found in store: {name}")
            if record.logical_length < self.window_size:
                continue
            count = 1 + (record.logical_length - self.window_size) // self.stride
            records.append((record.name, record.logical_length, count, record.byte_offset, record.byte_length))
        if not records:
            raise ValueError("No selected chromosome is long enough for window_size")

        self._records = records
        self._offsets = [0]
        for _, _, count, _, _ in records:
            self._offsets.append(self._offsets[-1] + count)
        self._group = None

    @property
    def chromosomes(self) -> tuple[str, ...]:
        return tuple(name for name, _, _, _, _ in self._records)

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_group"] = None
        state["_arrays"] = {}
        return state

    def _array(self, chromosome: str):
        if self._group is None:
            self._group = _open_group(self.store_path)
        array_key = self._array_name if self._layout == "flat-packed" else chromosome
        array = self._arrays.get(array_key)
        if array is None:
            array = self._group[array_key]
            self._arrays[array_key] = array
        return array

    def _location(self, index: int) -> tuple[str, int, int, int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("GenomeZarrDataset index out of range")
        record_index = bisect_right(self._offsets, index) - 1
        name, logical_length, _, byte_offset, byte_length = self._records[record_index]
        start = (index - self._offsets[record_index]) * self.stride
        return name, logical_length, start, byte_offset, byte_length

    def __getitem__(self, index: int):
        if not isinstance(index, Integral):
            raise TypeError("GenomeZarrDataset indices must be integers")
        chromosome, logical_length, start, byte_offset, byte_length = self._location(int(index))
        byte_start = byte_offset + (start // 2)
        byte_end = byte_offset + min(byte_length, (start + self.window_size + 1) // 2)
        packed = np.asarray(self._array(chromosome)[byte_start:byte_end], dtype=np.uint8)
        bases = _decode_packed(packed)
        sequence = bases[start % 2 : start % 2 + self.window_size]
        if sequence.size != self.window_size:
            raise RuntimeError(f"Store ended unexpectedly while reading {chromosome!r}")
        if self.return_metadata:
            return {"sequence": sequence, "chromosome": chromosome, "start": start, "end": start + self.window_size}
        return sequence


def create_dataloader(
    store_path: str | Path,
    window_size: int,
    *,
    batch_size: int = 1,
    stride: int | None = None,
    chromosomes: str | Sequence[str] | None = None,
    return_metadata: bool = False,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool | None = None,
    **dataloader_kwargs,
):
    if DataLoader is None:  # pragma: no cover - depends on caller environment
        raise RuntimeError("create_dataloader requires PyTorch; install torch first")
    if not isinstance(batch_size, Integral) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0")
    dataset = GenomeZarrDataset(
        store_path,
        window_size,
        stride=stride,
        chromosomes=chromosomes,
        return_metadata=return_metadata,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        **dataloader_kwargs,
    )


class ChunkedGenomeZarrDataset(IterableDataset):
    """Stream windows while decoding each packed chunk only once."""

    def __init__(
        self,
        store_path: str | Path,
        window_size: int,
        *,
        stride: int | None = None,
        chromosomes: str | Sequence[str] | None = None,
        shuffle: bool = False,
        seed: int = 0,
        return_metadata: bool = False,
    ) -> None:
        if not isinstance(window_size, Integral) or isinstance(window_size, bool) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if stride is None:
            stride = window_size
        if not isinstance(stride, Integral) or isinstance(stride, bool) or stride <= 0:
            raise ValueError("stride must be a positive integer")
        self.store_path = str(Path(store_path))
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.chromosomes = (chromosomes,) if isinstance(chromosomes, str) else tuple(chromosomes) if chromosomes else None
        self.shuffle = shuffle
        self.seed = int(seed)
        self.return_metadata = return_metadata

        # Parse the manifest / build the record index ONCE, in the constructing
        # (main) process. DataLoader workers then inherit this via fork instead
        # of each re-reading the manifest in __iter__. Only the lightweight
        # record metadata is kept here; the Zarr array handle is not picklable
        # across processes and is opened lazily per worker in __iter__.
        index = _load_store_index(self.store_path)
        self._layout = index.layout
        self._array_name = index.array_name
        self._records = index.records
        self.compressor = index.compressor

    def _item(self, chromosome: str, start: int, sequence: np.ndarray):
        if self.return_metadata:
            return {"sequence": sequence, "chromosome": chromosome, "start": start, "end": start + self.window_size}
        return sequence

    def __iter__(self):
        available = {record.name for record in self._records}
        if not available:
            raise ValueError("Store contains no chromosome arrays")
        names = sorted(available) if self.chromosomes is None else list(self.chromosomes)
        missing = sorted(set(names).difference(available))
        if missing:
            raise KeyError("Chromosomes not found in store: " + ", ".join(missing))

        worker = get_worker_info() if get_worker_info is not None else None
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        record_rng = random.Random(self.seed)
        window_rng = random.Random(self.seed + worker_id)
        if self.shuffle:
            record_rng.shuffle(names)
        names = names[worker_id::worker_count]

        # Open the Zarr array lazily, once per worker. The record index was
        # parsed in __init__ and inherited here, so no worker re-reads the
        # manifest.
        group = _open_group(self.store_path)
        record_map = {record.name: record for record in self._records}
        if self._layout == "flat-packed":
            array = group[self._array_name]
            chunk_bytes = int(array.chunks[0] if array.chunks else array.shape[0])
            for chromosome in names:
                record = record_map[chromosome]
                if record.logical_length < self.window_size:
                    continue
                record_start = record.byte_offset
                record_end = record_start + record.byte_length
                max_start = record.logical_length - self.window_size
                for byte_start in range(record_start, record_end, chunk_bytes):
                    byte_end = min(byte_start + chunk_bytes, record_end)
                    base_start = (byte_start - record_start) * 2
                    base_end = min((byte_end - record_start) * 2, record.logical_length)
                    first_start = ((base_start + self.stride - 1) // self.stride) * self.stride
                    starts = list(range(first_start, min(base_end, max_start + 1), self.stride))
                    if self.shuffle:
                        window_rng.shuffle(starts)
                    decoded = _decode_packed(np.asarray(array[byte_start:byte_end], dtype=np.uint8))
                    for start in starts:
                        local_start = start - base_start
                        local_end = local_start + self.window_size
                        if local_end <= decoded.size:
                            yield self._item(chromosome, start, decoded[local_start:local_end].copy())
                        else:
                            end_byte = record_start + (start + self.window_size + 1) // 2
                            packed = np.asarray(array[record_start + (start // 2):end_byte], dtype=np.uint8)
                            bases = _decode_packed(packed)
                            offset = start % 2
                            yield self._item(chromosome, start, bases[offset:offset + self.window_size].copy())
        else:
            for chromosome in names:
                array = group[chromosome]
                if array.ndim != 1 or np.dtype(array.dtype) != np.dtype(np.uint8):
                    raise ValueError(f"Array {chromosome!r} is not a one-dimensional uint8 packed chromosome")
                logical_length = int(array.attrs.get("logical_length", int(array.shape[0]) * 2))
                if logical_length < self.window_size:
                    continue
                chunk_bytes = int(array.chunks[0])
                max_start = logical_length - self.window_size
                for byte_start in range(0, int(array.shape[0]), chunk_bytes):
                    byte_end = min(byte_start + chunk_bytes, int(array.shape[0]))
                    base_start = byte_start * 2
                    base_end = min(byte_end * 2, logical_length)
                    first_start = ((base_start + self.stride - 1) // self.stride) * self.stride
                    starts = list(range(first_start, min(base_end, max_start + 1), self.stride))
                    if self.shuffle:
                        window_rng.shuffle(starts)
                    decoded = _decode_packed(np.asarray(array[byte_start:byte_end], dtype=np.uint8))
                    for start in starts:
                        local_start = start - base_start
                        local_end = local_start + self.window_size
                        if local_end <= decoded.size:
                            yield self._item(chromosome, start, decoded[local_start:local_end].copy())
                        else:
                            end_byte = (start + self.window_size + 1) // 2
                            packed = np.asarray(array[start // 2:end_byte], dtype=np.uint8)
                            bases = _decode_packed(packed)
                            offset = start % 2
                            yield self._item(chromosome, start, bases[offset:offset + self.window_size].copy())


def create_chunked_dataloader(
    store_path: str | Path,
    window_size: int,
    *,
    batch_size: int = 1,
    stride: int | None = None,
    chromosomes: str | Sequence[str] | None = None,
    return_metadata: bool = False,
    shuffle: bool = False,
    seed: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool | None = None,
    **dataloader_kwargs,
):
    if DataLoader is None:  # pragma: no cover - depends on caller environment
        raise RuntimeError("create_chunked_dataloader requires PyTorch; install torch first")
    if not isinstance(batch_size, Integral) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0")
    dataset = ChunkedGenomeZarrDataset(
        store_path,
        window_size,
        stride=stride,
        chromosomes=chromosomes,
        return_metadata=return_metadata,
        shuffle=shuffle,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        **dataloader_kwargs,
    )
