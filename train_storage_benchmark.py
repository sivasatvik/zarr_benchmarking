#!/usr/bin/env python3
"""Train the same small next-base model from indexed FASTA and packed Zarr.

The initial ``--build-manifest`` pass is deliberately excluded from training
timings. It creates a deterministic sample of windows and resolves their byte
locations in both formats, making later backend runs directly comparable.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np


DEFAULT_FASTA = "/gpfs/scratch/sm12779/opengenome2/fasta/plasmids_phage/imgvr.fasta"
# DEFAULT_ZARR = "/gpfs/data/jt3545lab/home/sm12779/genome-zarr-data/opengenome2_store/imgvr/imgvr.zarr"
DEFAULT_ZARR = "/gpfs/scratch/sm12779/opengenome2_store/imgvr_uncompressed.zarr"
# DEFAULT_ZARR = "/gpfs/scratch/sm12779/opengenome2_store/imgvr.zarr"

_CODES = np.full(256, 4, dtype=np.uint8)
for _base, _code in ((b"A", 0), (b"C", 1), (b"G", 2), (b"T", 3), (b"N", 4)):
    _CODES[_base[0]] = _code
    _CODES[_base.lower()[0]] = _code

# Explicit, stable CSV layout. Adding metadata and an isolated loading pass
# changes the schema, so results now record which store, compressor, chunk
# size, and storage tier produced each row instead of relying on a hand-edited
# backend label.
FIELDNAMES = [
    "backend", "tier", "store", "compressor", "chunk_bytes",
    "load_first_batch_seconds", "load_seconds", "load_bases", "load_bases_per_second",
    "first_batch_seconds", "train_seconds", "batches", "bases", "bases_per_second", "final_loss",
    "dataset_construction_seconds", "window_size", "batch_size", "steps_requested",
    "num_workers", "device", "manifest",
]


def build_fai(fasta: Path, fai: Path) -> None:
    """Build a standard fixed-line FASTA index without an external dependency."""
    print(f"Building FASTA index (one full scan): {fai}", flush=True)
    temporary = fai.with_suffix(fai.suffix + ".tmp")
    with fasta.open("rb") as source, temporary.open("wt", encoding="utf-8") as output:
        name = None
        sequence_offset = length = bases_per_line = bytes_per_line = 0
        saw_short_line = False
        while True:
            line_offset = source.tell()
            line = source.readline()
            if not line:
                if name is not None:
                    output.write(f"{name}\t{length}\t{sequence_offset}\t{bases_per_line}\t{bytes_per_line}\n")
                break
            if line.startswith(b">"):
                if name is not None:
                    output.write(f"{name}\t{length}\t{sequence_offset}\t{bases_per_line}\t{bytes_per_line}\n")
                name = line[1:].split(None, 1)[0].decode("ascii")
                sequence_offset = length = bases_per_line = bytes_per_line = 0
                saw_short_line = False
                continue
            if name is None or not line.strip():
                continue
            bases = len(line.rstrip(b"\r\n"))
            if bases == 0:
                continue
            if bases_per_line == 0:
                sequence_offset = line_offset
                bases_per_line, bytes_per_line = bases, len(line)
            elif saw_short_line or (bases != bases_per_line and len(line) == bytes_per_line):
                raise ValueError("FASTA has irregular line wrapping; build it with samtools faidx instead")
            if bases < bases_per_line:
                saw_short_line = True
            length += bases
    temporary.replace(fai)


def read_fai(fai: Path) -> dict[str, tuple[int, int, int, int]]:
    records = {}
    with fai.open("rt", encoding="utf-8") as handle:
        for line in handle:
            name, length, offset, bases_per_line, bytes_per_line, *_ = line.rstrip("\n").split("\t")
            records[name] = (int(length), int(offset), int(bases_per_line), int(bytes_per_line))
    return records


def build_manifest(zarr_store: Path, fasta: Path, fai: Path, output: Path, sequence_length: int, samples: int, seed: int) -> None:
    """Create a sampled window manifest shared by the two storage backends."""
    from genome_zarr.dataloader import GenomeZarrDataset

    if not fai.exists():
        build_fai(fasta, fai)
    print("Loading Zarr record manifest and computing global indexed windows...", flush=True)
    dataset = GenomeZarrDataset(zarr_store, sequence_length)
    print(f"Sampling {samples:,} of {len(dataset):,} valid Zarr windows...", flush=True)
    rng = np.random.default_rng(seed)
    fasta_records = read_fai(fai)

    names, starts, zarr_offsets, zarr_lengths = [], [], [], []
    fasta_offsets, fasta_bases, fasta_bytes = [], [], []
    skipped_missing = skipped_short = 0
    # The conversion may have disambiguated duplicate FASTA headers by adding
    # ``_duplicate_N``. Such Zarr-only records have no direct FASTA baseline,
    # so skip them and continue sampling shared records instead.
    while len(names) < samples:
        remaining = samples - len(names)
        candidate_indices = rng.integers(0, len(dataset), size=max(remaining * 2, 1024), dtype=np.int64)
        for index in candidate_indices:
            name, _, start, byte_offset, byte_length = dataset._location(int(index))
            fasta_record = fasta_records.get(name)
            if fasta_record is None:
                skipped_missing += 1
                continue
            fasta_length, fasta_offset, bases_per_line, bytes_per_line = fasta_record
            if start + sequence_length > fasta_length:
                skipped_short += 1
                continue
            names.append(name)
            starts.append(start)
            zarr_offsets.append(byte_offset)
            zarr_lengths.append(byte_length)
            fasta_offsets.append(fasta_offset)
            fasta_bases.append(bases_per_line)
            fasta_bytes.append(bytes_per_line)
            if len(names) == samples:
                break
        if not names:
            raise ValueError("No Zarr records were shared with the FASTA index")
    print(
        f"Selected {len(names):,} shared windows; skipped {skipped_missing:,} Zarr-only records "
        f"and {skipped_short:,} length-mismatched windows.",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        version=np.array(1), sequence_length=np.array(sequence_length), names=np.asarray(names),
        starts=np.asarray(starts, dtype=np.int64), zarr_offsets=np.asarray(zarr_offsets, dtype=np.int64),
        zarr_lengths=np.asarray(zarr_lengths, dtype=np.int64), fasta_offsets=np.asarray(fasta_offsets, dtype=np.int64),
        fasta_bases_per_line=np.asarray(fasta_bases, dtype=np.int32), fasta_bytes_per_line=np.asarray(fasta_bytes, dtype=np.int32),
    )
    print(f"Wrote shared manifest: {output}", flush=True)


class _WindowDataset:
    """Shared sampled window metadata; subclasses only differ in byte reads."""

    def __init__(self, manifest: Path):
        data = np.load(manifest, allow_pickle=False)
        self.sequence_length = int(data["sequence_length"])
        self.starts = data["starts"]
        self.zarr_offsets = data["zarr_offsets"]
        self.zarr_lengths = data["zarr_lengths"]
        self.fasta_offsets = data["fasta_offsets"]
        self.fasta_bases_per_line = data["fasta_bases_per_line"]
        self.fasta_bytes_per_line = data["fasta_bytes_per_line"]

    def __len__(self):
        return len(self.starts)


def _dataset_classes():
    """Delay torch import so ``--build-manifest`` needs only Zarr and NumPy."""
    import torch
    from torch.utils.data import Dataset
    import zarr

    class FastaWindowDataset(_WindowDataset, Dataset):
        def __init__(self, fasta: Path, manifest: Path):
            Dataset.__init__(self)
            _WindowDataset.__init__(self, manifest)
            self.fasta = str(fasta)
            self._handle = None

        def __getstate__(self):
            state = self.__dict__.copy()
            state["_handle"] = None
            return state

        def __getitem__(self, index):
            if self._handle is None:
                self._handle = open(self.fasta, "rb", buffering=0)
            start = int(self.starts[index])
            bases_per_line = int(self.fasta_bases_per_line[index])
            bytes_per_line = int(self.fasta_bytes_per_line[index])
            first = int(self.fasta_offsets[index]) + (start // bases_per_line) * bytes_per_line + start % bases_per_line
            end = start + self.sequence_length
            last = int(self.fasta_offsets[index]) + (end // bases_per_line) * bytes_per_line + end % bases_per_line
            self._handle.seek(first)
            raw = self._handle.read(last - first).replace(b"\n", b"").replace(b"\r", b"")
            if len(raw) != self.sequence_length:
                raise RuntimeError("FASTA index produced a short sequence read")
            return torch.from_numpy(_CODES[np.frombuffer(raw, dtype=np.uint8)].copy())

    class ZarrWindowDataset(_WindowDataset, Dataset):
        def __init__(self, store: Path, manifest: Path):
            Dataset.__init__(self)
            _WindowDataset.__init__(self, manifest)
            self.store = str(store)
            self._array = None

        def __getstate__(self):
            state = self.__dict__.copy()
            state["_array"] = None
            return state

        def __getitem__(self, index):
            if self._array is None:
                self._array = zarr.open_group(self.store, mode="r")["packed_sequence"]
            start = int(self.starts[index])
            record_offset = int(self.zarr_offsets[index])
            record_length = int(self.zarr_lengths[index])
            byte_start = record_offset + start // 2
            byte_end = record_offset + min(record_length, (start + self.sequence_length + 1) // 2)
            packed = np.asarray(self._array[byte_start:byte_end], dtype=np.uint8)
            bases = np.empty(packed.size * 2, dtype=np.uint8)
            bases[0::2], bases[1::2] = packed >> 4, packed & 15
            sequence = bases[start % 2:start % 2 + self.sequence_length]
            if sequence.size != self.sequence_length:
                raise RuntimeError("Zarr produced a short sequence read")
            return torch.from_numpy(sequence.copy())

    return FastaWindowDataset, ZarrWindowDataset


def verify_backends(fasta: Path, zarr_store: Path, manifest: Path, count: int, classes) -> None:
    """Fail early if either backend yields a different encoded sequence."""
    if count <= 0:
        return
    FastaWindowDataset, ZarrWindowDataset = classes
    fasta_dataset = FastaWindowDataset(fasta, manifest)
    zarr_dataset = ZarrWindowDataset(zarr_store, manifest)
    for index in range(min(count, len(fasta_dataset))):
        if not np.array_equal(fasta_dataset[index].numpy(), zarr_dataset[index].numpy()):
            raise RuntimeError(f"FASTA and Zarr differ for sampled manifest row {index}")
    print(f"Verified identical FASTA and Zarr base codes for {min(count, len(fasta_dataset))} sampled windows.", flush=True)


def model(sequence_length: int, device):
    import torch

    class NextBaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(5, 16)
            self.network = torch.nn.Sequential(
                torch.nn.Conv1d(16, 32, kernel_size=5, padding=2),
                torch.nn.GELU(),
                torch.nn.Conv1d(32, 5, kernel_size=1),
            )

        def forward(self, inputs):
            return self.network(self.embedding(inputs).transpose(1, 2))

    return NextBaseModel().to(device)


def train_backend(name, dataset, args, device) -> dict:
    import torch
    from torch.utils.data import DataLoader

    # Make the model initialization and DataLoader permutation identical for
    # FASTA and Zarr; any loss difference then signals unequal input data.
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.num_workers > 0)
    network = model(dataset.sequence_length, device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=4)
    iterator = iter(loader)
    start = time.perf_counter()
    first_batch = next(iterator)
    first_batch_seconds = time.perf_counter() - start
    if device.type == "cuda":
        torch.cuda.synchronize()
    started_training = time.perf_counter()
    loss_value = 0.0
    bases = 0
    batches = 0
    batch = first_batch
    for step in range(args.steps):
        # Both storage datasets return compact uint8 base codes. Embedding and
        # cross-entropy targets require integer indices, so convert only after
        # the host-to-device transfer rather than inflating worker batches.
        batch = batch.to(device, dtype=torch.long, non_blocking=True)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits = network(inputs).transpose(1, 2)
        loss = criterion(logits.reshape(-1, 5), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        bases += int(targets.numel())
        batches += 1
        if step + 1 < args.steps:
            try:
                batch = next(iterator)
            except StopIteration:
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started_training
    return {"backend": name, "first_batch_seconds": first_batch_seconds, "train_seconds": train_seconds,
            "batches": batches, "bases": bases, "bases_per_second": bases / train_seconds if train_seconds else 0.0,
            "final_loss": loss_value}


def store_metadata(store: Path) -> tuple[str, str]:
    """Return ``(compressor_name, chunk_bytes)`` recorded in a packed Zarr store.

    This lets each CSV row document the store it measured, so a compressed and
    an uncompressed run are distinguishable without hand-editing the backend
    label.
    """
    import zarr

    group = zarr.open_group(str(store), mode="r")
    array = group["packed_sequence"]
    chunks = getattr(array, "chunks", None)
    chunk_bytes = str(int(chunks[0])) if chunks else str(int(array.shape[0]))
    compressor = group.attrs.get("compressor")
    if not compressor:
        try:
            codec_names = [type(codec).__name__.lower() for codec in array.metadata.codecs]
            compressor = "zstd" if any("zstd" in name for name in codec_names) else "none"
        except Exception:
            compressor = "unknown"
    return str(compressor), chunk_bytes


def measure_loading(dataset, args) -> dict:
    """Time random-window delivery without the model to isolate read throughput.

    It mirrors ``train_backend``'s loader configuration and batch/base
    accounting so the figure is directly comparable to the training figure,
    but performs no host-to-device copy and runs no model. The result is the
    storage + decode + collate cost of the random 4 KB-window access pattern.
    """
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
                        num_workers=args.num_workers, pin_memory=False,
                        persistent_workers=args.num_workers > 0)
    iterator = iter(loader)
    start = time.perf_counter()
    batch = next(iterator)
    first_batch_seconds = time.perf_counter() - start
    bases = 0
    batches = 0
    started = time.perf_counter()
    for step in range(args.steps):
        # The loader has already produced the batch in this process; counting
        # its elements is enough to keep the timing about delivery, not compute.
        bases += int(batch[:, 1:].numel())
        batches += 1
        if step + 1 < args.steps:
            try:
                batch = next(iterator)
            except StopIteration:
                break
    total_seconds = first_batch_seconds + (time.perf_counter() - started)
    return {"load_first_batch_seconds": first_batch_seconds, "load_seconds": total_seconds,
            "load_bases": bases, "load_bases_per_second": bases / total_seconds if total_seconds else 0.0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-store", type=Path, default=Path(DEFAULT_ZARR))
    parser.add_argument("--fasta", type=Path, default=Path(DEFAULT_FASTA))
    parser.add_argument("--fasta-index", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=Path("imgvr_windows_1025_100000.npz"))
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--setup-only", action="store_true", help="Create/rebuild the FASTA index and sampled manifest, then exit")
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--window-size", type=int, default=1024, help="Model input bases; one next-base target is added")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--verify-samples", type=int, default=32, help="Compare this many shared windows before training; 0 disables")
    parser.add_argument("--backend", choices=("fasta", "zarr", "both"), default="both")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--results", type=Path, default=Path("storage_training_benchmark.csv"))
    parser.add_argument("--tier", default="", help="Storage-tier label recorded in the CSV, e.g. nvme, gpfs-ssd, gpfs-hdd")
    parser.add_argument("--loading-only", action="store_true", help="Measure only random-window loading throughput; skip model training")
    args = parser.parse_args(argv)
    args.fasta_index = args.fasta_index or Path(str(args.fasta) + ".fai")
    sequence_length = args.window_size + 1
    if args.build_manifest or not args.manifest.exists():
        build_manifest(args.zarr_store, args.fasta, args.fasta_index, args.manifest, sequence_length, args.samples, args.seed)
    data = np.load(args.manifest, allow_pickle=False)
    if int(data["sequence_length"]) != sequence_length:
        parser.error("Manifest window size differs; pass --build-manifest with a new --manifest path")
    if args.setup_only:
        print(f"Setup complete: {args.manifest}")
        return

    import torch
    cuda_ready = torch.cuda.is_available()
    if not args.loading_only and args.device == "cuda" and not cuda_ready:
        parser.error("CUDA was requested but is unavailable; use --device cpu")
    # A loading-only run never touches the GPU, so it can proceed on CPU even
    # where CUDA was requested but is unavailable.
    device = torch.device("cpu" if (args.device == "cuda" and not cuda_ready) else args.device)
    torch.manual_seed(args.seed)

    if args.results.exists():
        with args.results.open("rt", newline="", encoding="utf-8") as handle:
            existing_header = handle.readline().rstrip("\r\n").split(",")
        if existing_header and existing_header != FIELDNAMES:
            parser.error(
                f"{args.results} was written with a different column layout; "
                "pass a new --results path for the updated schema"
            )
    FastaWindowDataset, ZarrWindowDataset = _dataset_classes()
    verify_backends(args.fasta, args.zarr_store, args.manifest, args.verify_samples, (FastaWindowDataset, ZarrWindowDataset))
    zarr_compressor, zarr_chunk_bytes = "", ""
    try:
        zarr_compressor, zarr_chunk_bytes = store_metadata(args.zarr_store)
    except Exception as exc:  # pragma: no cover - depends on the store contents
        print(f"Warning: could not read Zarr store metadata: {exc}", flush=True)

    def make_dataset(backend):
        if backend == "fasta":
            return FastaWindowDataset(args.fasta, args.manifest)
        return ZarrWindowDataset(args.zarr_store, args.manifest)

    results = []
    for backend in (("fasta",) if args.backend == "fasta" else ("zarr",) if args.backend == "zarr" else ("fasta", "zarr")):
        created = time.perf_counter()
        dataset = make_dataset(backend)
        construction_seconds = time.perf_counter() - created

        # Isolated random-window loading pass: storage + decode + collate only,
        # no model and no device transfer, so the throughput is a clean read
        # figure rather than an end-to-end training figure.
        load_row = measure_loading(dataset, args)

        if args.loading_only:
            row = {"backend": backend, "first_batch_seconds": "", "train_seconds": "",
                   "batches": "", "bases": "", "bases_per_second": "", "final_loss": ""}
        else:
            row = train_backend(backend, make_dataset(backend), args, device)

        if backend == "zarr":
            store, compressor, chunk_bytes = str(args.zarr_store), zarr_compressor, zarr_chunk_bytes
        else:
            store, compressor, chunk_bytes = str(args.fasta), "", ""

        row.update(load_row)
        row.update({"dataset_construction_seconds": construction_seconds, "window_size": args.window_size,
                    "batch_size": args.batch_size, "steps_requested": args.steps, "num_workers": args.num_workers,
                    "device": str(device), "store": store, "compressor": compressor, "chunk_bytes": chunk_bytes,
                    "tier": args.tier, "manifest": str(args.manifest)})
        results.append(row)
        print("{backend} [{compressor}] loading: first batch {load_first_batch_seconds:.3f}s; "
              "{load_seconds:.3f}s total; {load_bases_per_second:,.0f} bases/s".format(**row), flush=True)
        if not args.loading_only:
            print("{backend} training: {train_seconds:.3f}s; {bases_per_second:,.0f} bases/s; loss {final_loss:.4f}".format(**row), flush=True)

    write_header = not args.results.exists()
    with args.results.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(results)
    print(f"Appended results to {args.results}")


if __name__ == "__main__":
    main()
