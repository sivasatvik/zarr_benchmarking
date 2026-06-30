#!/usr/bin/env python3
"""
Benchmark compression algorithms across genome encodings and Zarr chunk sizes.

This script intentionally uses the same file-backed, Zarr-v3-style chunk layout
as the other scripts in this folder, so it does not require the Python zarr
package. Each benchmark store has:

  <output>/stores/<architecture>/<chunk_size>/<compressor>.zarr/
      zarr.json
      c/<chunk files>

For every architecture/chunk/compressor combination it can:
  1. build compressed chunk files from a FASTA,
  2. read random windows and report throughput + latency,
  3. report apparent and physical storage,
  4. write CSV results and optional PNG plots.

Example:
  python zarr_compression_benchmark.py \
    --fasta ./hg38_data/hg38.fa \
    --output-dir ./hg38_benchmark_data/compressed_zarr_benchmark \
    --chromosomes chr1 \
    --compressors none zlib gzip bz2 lzma \
    --chunk-sizes 128KB 256KB 512KB 1MB 2MB 4MB 16MB
"""

import argparse
import bz2
import csv
import gzip
import json
import lzma
import math
import os
import re
import shutil
import statistics
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

import numpy as np


DEFAULT_CHUNK_SIZES = {
    "128KB": 131_072,
    "256KB": 262_144,
    "512KB": 524_288,
    "1MB": 1_048_576,
    "2MB": 2_097_152,
    "4MB": 4_194_304,
    "16MB": 16_777_216,
}
DEFAULT_ARCHITECTURES = ("uint8", "4bit", "2bit_flag")
DEFAULT_COMPRESSORS = ("none", "zlib", "gzip", "lz4", "zstd") #"bz2", "lzma", )
BASES_PER_MIB = 1024 * 1024


class Chromosome(NamedTuple):
    name: str
    length: int


class Request(NamedTuple):
    chrom: str
    start: int
    bases: int


class Compressor(object):
    def __init__(self, name):
        self.name = name

    def compress(self, data):
        if self.name == "none":
            return data
        if self.name == "zlib":
            return zlib.compress(data, level=6)
        if self.name == "gzip":
            return gzip.compress(data, compresslevel=6)
        if self.name == "bz2":
            return bz2.compress(data, compresslevel=6)
        if self.name == "lzma":
            return lzma.compress(data, preset=6)
        if self.name == "lz4":
            import lz4.frame

            return lz4.frame.compress(data, compression_level=0)
        if self.name == "zstd":
            import zstandard

            return zstandard.ZstdCompressor(level=3).compress(data)
        raise ValueError("Unknown compressor: {}".format(self.name))

    def decompress(self, data):
        if self.name == "none":
            return data
        if self.name == "zlib":
            return zlib.decompress(data)
        if self.name == "gzip":
            return gzip.decompress(data)
        if self.name == "bz2":
            return bz2.decompress(data)
        if self.name == "lzma":
            return lzma.decompress(data)
        if self.name == "lz4":
            import lz4.frame

            return lz4.frame.decompress(data)
        if self.name == "zstd":
            import zstandard

            return zstandard.ZstdDecompressor().decompress(data)
        raise ValueError("Unknown compressor: {}".format(self.name))


def parse_size(label):
    match = re.fullmatch(r"(\d+)([KMG]?B)?", label.strip().upper())
    if not match:
        raise argparse.ArgumentTypeError("Invalid size: {}".format(label))
    value = int(match.group(1))
    unit = match.group(2) or "B"
    scale = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}[unit]
    canonical = "{}{}".format(value, unit) if unit != "B" else str(value)
    return canonical, value * scale


def validate_compressor(name):
    if name in ("none", "zlib", "gzip", "bz2", "lzma"):
        return
    if name == "lz4":
        try:
            import lz4.frame  # noqa: F401
        except ImportError:
            raise SystemExit("Compressor 'lz4' needs: pip install lz4")
        return
    if name == "zstd":
        try:
            import zstandard  # noqa: F401
        except ImportError:
            raise SystemExit("Compressor 'zstd' needs: pip install zstandard")
        return
    raise SystemExit("Unknown compressor: {}".format(name))


def fasta_index_path(fasta):
    return fasta.with_name(fasta.name + ".fai")


def read_fai(fasta):
    fai = fasta_index_path(fasta)
    if not fai.exists():
        return None
    chromosomes = []
    with fai.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 2:
                chromosomes.append(Chromosome(fields[0], int(fields[1])))
    return chromosomes


def iter_fasta_records(fasta, wanted):
    # type: (Path, Optional[Set[str]]) -> Iterable[Tuple[str, str]]
    name = None
    parts = []
    with fasta.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None and (wanted is None or name in wanted):
                    yield name, "".join(parts)
                name = line[1:].split()[0]
                parts = []
            elif name is not None and (wanted is None or name in wanted):
                parts.append(line)
        if name is not None and (wanted is None or name in wanted):
            yield name, "".join(parts)


def select_chromosomes(fasta, chromosome_args):
    indexed = read_fai(fasta)
    if chromosome_args == ["all"]:
        if indexed is not None:
            return indexed
        return [Chromosome(name, len(seq)) for name, seq in iter_fasta_records(fasta, None)]

    wanted = set(chromosome_args)
    if indexed is not None:
        found = [chrom for chrom in indexed if chrom.name in wanted]
        missing = wanted.difference(set(chrom.name for chrom in found))
        if missing:
            raise SystemExit("Chromosome(s) missing from FASTA index: {}".format(", ".join(sorted(missing))))
        return found

    records = [Chromosome(name, len(seq)) for name, seq in iter_fasta_records(fasta, wanted)]
    missing = wanted.difference(set(chrom.name for chrom in records))
    if missing:
        raise SystemExit("Chromosome(s) missing from FASTA: {}".format(", ".join(sorted(missing))))
    return records


def encode_uint8(seq):
    mapping = np.full(256, 4, dtype=np.uint8)
    for base, code in {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}.items():
        mapping[ord(base)] = code
        mapping[ord(base.lower())] = code
    return mapping[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)].tobytes()


def encode_4bit(seq):
    values = np.frombuffer(encode_uint8(seq), dtype=np.uint8)
    values = np.where(values > 3, 4, values).astype(np.uint8)
    if values.size % 2:
        values = np.pad(values, (0, 1), constant_values=0)
    return ((values[0::2] << 4) | values[1::2]).astype(np.uint8).tobytes()


def encode_2bit_flag(seq):
    values = np.frombuffer(encode_uint8(seq), dtype=np.uint8)
    flags = (values == 4).astype(np.uint8)
    bases = np.where(values > 3, 0, values).astype(np.uint8)

    if bases.size % 4:
        bases = np.pad(bases, (0, 4 - bases.size % 4), constant_values=0)
    seq_bytes = (
        (bases[0::4] << 6)
        | (bases[1::4] << 4)
        | (bases[2::4] << 2)
        | bases[3::4]
    ).astype(np.uint8).tobytes()

    if flags.size % 8:
        flags = np.pad(flags, (0, 8 - flags.size % 8), constant_values=0)
    return seq_bytes, np.packbits(flags, bitorder="big").tobytes()


def bytes_for_bases(arch, logical_bases, component="data"):
    if arch == "uint8":
        return logical_bases
    if arch == "4bit":
        return int(math.ceil(logical_bases / 2.0))
    if arch == "2bit_flag" and component == "seq":
        return int(math.ceil(logical_bases / 4.0))
    if arch == "2bit_flag" and component == "flag":
        return int(math.ceil(logical_bases / 8.0))
    raise ValueError("Unsupported arch/component: {}/{}".format(arch, component))


def encoded_length(arch, bases, component="data"):
    return bytes_for_bases(arch, bases, component)


def store_path(root, arch, chunk_name, compressor_name):
    return root / "stores" / arch / chunk_name / "{}.zarr".format(compressor_name)


def write_zarr_metadata(store, arch, chunk_name, chunk_bases, compressor_name, chromosomes):
    meta = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "benchmark_layout": "manual_chunked",
            "architecture": arch,
            "chunk_name": chunk_name,
            "logical_chunk_bases": chunk_bases,
            "compressor": compressor_name,
            "chromosomes": [{"name": c.name, "length": c.length} for c in chromosomes],
        },
    }
    with (store / "zarr.json").open("w") as handle:
        json.dump(meta, handle, indent=2)


def chunk_file_name(arch, chrom, component, index):
    if arch == "2bit_flag":
        return "{}_{}_{}".format(chrom, component, index)
    return "{}_{}".format(chrom, index)


def write_compressed_chunks(chunk_dir, arch, chrom, payload, chunk_len, compressor, component="data"):
    num_chunks = int(math.ceil(len(payload) / float(chunk_len))) if payload else 0
    for idx in range(num_chunks):
        start = idx * chunk_len
        chunk = payload[start : start + chunk_len]
        out = chunk_dir / chunk_file_name(arch, chrom, component, idx)
        with out.open("wb") as handle:
            handle.write(compressor.compress(chunk))


def build_store(fasta, out_store, arch, chunk_name, chunk_bases, compressor_name, chromosomes, force):
    done_marker = out_store / ".benchmark_complete.json"
    if done_marker.exists() and not force:
        return 0.0

    if out_store.exists():
        shutil.rmtree(str(out_store))
    chunk_dir = out_store / "c"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    write_zarr_metadata(out_store, arch, chunk_name, chunk_bases, compressor_name, chromosomes)

    wanted = set(chrom.name for chrom in chromosomes)
    compressor = Compressor(compressor_name)
    start_time = time.perf_counter()

    for chrom, seq in iter_fasta_records(fasta, wanted):
        if arch == "uint8":
            write_compressed_chunks(
                chunk_dir, arch, chrom, encode_uint8(seq), bytes_for_bases(arch, chunk_bases), compressor
            )
        elif arch == "4bit":
            write_compressed_chunks(
                chunk_dir, arch, chrom, encode_4bit(seq), bytes_for_bases(arch, chunk_bases), compressor
            )
        elif arch == "2bit_flag":
            seq_bytes, flag_bytes = encode_2bit_flag(seq)
            write_compressed_chunks(
                chunk_dir, arch, chrom, seq_bytes, bytes_for_bases(arch, chunk_bases, "seq"), compressor, "seq"
            )
            write_compressed_chunks(
                chunk_dir, arch, chrom, flag_bytes, bytes_for_bases(arch, chunk_bases, "flag"), compressor, "flag"
            )
        else:
            raise ValueError("Unsupported architecture: {}".format(arch))

    build_seconds = time.perf_counter() - start_time
    with done_marker.open("w") as handle:
        json.dump({"build_seconds": build_seconds}, handle, indent=2)
    return build_seconds


def generate_requests(chromosomes, num_requests, window_bases, seed):
    eligible = [chrom for chrom in chromosomes if chrom.length > window_bases]
    if not eligible:
        raise SystemExit("No selected chromosome is longer than --window-bases.")
    rng = np.random.RandomState(seed)
    weights = np.array([chrom.length for chrom in eligible], dtype=np.float64)
    weights /= weights.sum()
    chrom_indices = rng.choice(len(eligible), size=num_requests, p=weights)

    requests = []
    for idx in chrom_indices:
        chrom = eligible[int(idx)]
        start = int(rng.randint(0, chrom.length - window_bases))
        requests.append(Request(chrom.name, start, window_bases))
    return requests


def read_chunk(chunk_dir, arch, chrom, component, index, compressor, drop_page_cache):
    path = chunk_dir / chunk_file_name(arch, chrom, component, index)
    with path.open("rb") as handle:
        data = handle.read()
        if drop_page_cache and hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            try:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
        return compressor.decompress(data)


def read_encoded_window(
    chunk_dir,
    arch,
    chrom,
    start_byte,
    end_byte,
    chunk_len,
    compressor,
    drop_page_cache,
    component="data",
):
    first_chunk = start_byte // chunk_len
    last_chunk = (end_byte - 1) // chunk_len
    pieces = []
    for idx in range(first_chunk, last_chunk + 1):
        chunk = read_chunk(chunk_dir, arch, chrom, component, idx, compressor, drop_page_cache)
        chunk_start = idx * chunk_len
        local_start = max(0, start_byte - chunk_start)
        local_end = min(len(chunk), end_byte - chunk_start)
        pieces.append(chunk[local_start:local_end])
    return b"".join(pieces)


def read_window(store, arch, chunk_bases, compressor, request, drop_page_cache):
    chunk_dir = store / "c"
    start = request.start
    end = request.start + request.bases

    if arch == "uint8":
        read_encoded_window(
            chunk_dir,
            arch,
            request.chrom,
            start,
            end,
            bytes_for_bases(arch, chunk_bases),
            compressor,
            drop_page_cache,
        )
    elif arch == "4bit":
        byte_start = start // 2
        byte_end = int(math.ceil(end / 2.0))
        read_encoded_window(
            chunk_dir,
            arch,
            request.chrom,
            byte_start,
            byte_end,
            bytes_for_bases(arch, chunk_bases),
            compressor,
            drop_page_cache,
        )
    elif arch == "2bit_flag":
        seq_start = start // 4
        seq_end = int(math.ceil(end / 4.0))
        flag_start = start // 8
        flag_end = int(math.ceil(end / 8.0))
        read_encoded_window(
            chunk_dir,
            arch,
            request.chrom,
            seq_start,
            seq_end,
            bytes_for_bases(arch, chunk_bases, "seq"),
            compressor,
            drop_page_cache,
            "seq",
        )
        read_encoded_window(
            chunk_dir,
            arch,
            request.chrom,
            flag_start,
            flag_end,
            bytes_for_bases(arch, chunk_bases, "flag"),
            compressor,
            drop_page_cache,
            "flag",
        )
    else:
        raise ValueError("Unsupported architecture: {}".format(arch))


def percentile(values, pct):
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values), pct))


def benchmark_store(path, arch, chunk_bases, compressor_name, requests, warmup, drop_page_cache):
    compressor = Compressor(compressor_name)
    for request in requests[:warmup]:
        read_window(path, arch, chunk_bases, compressor, request, drop_page_cache)

    latencies = []
    total_bases = 0
    wall_start = time.perf_counter()
    for request in requests:
        req_start = time.perf_counter()
        read_window(path, arch, chunk_bases, compressor, request, drop_page_cache)
        latencies.append(time.perf_counter() - req_start)
        total_bases += request.bases

    wall_seconds = time.perf_counter() - wall_start
    lat_ms = [value * 1000 for value in latencies]
    return {
        "throughput_mib_s": (total_bases / float(BASES_PER_MIB)) / wall_seconds,
        "avg_latency_ms": statistics.mean(lat_ms),
        "median_latency_ms": statistics.median(lat_ms),
        "p95_latency_ms": percentile(lat_ms, 95),
        "p99_latency_ms": percentile(lat_ms, 99),
        "wall_seconds": wall_seconds,
    }


def storage_bytes(path):
    apparent = 0
    physical = 0
    file_count = 0
    for root, _, files in os.walk(str(path)):
        for name in files:
            full = Path(root) / name
            stat = full.stat()
            apparent += stat.st_size
            physical += stat.st_blocks * 512
            file_count += 1
    return apparent, physical, file_count


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_plots(results_csv, output_dir):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError:
        print("Skipping plots because pandas/matplotlib/seaborn are not installed.", flush=True)
        return

    df = pd.read_csv(results_csv)
    if df.empty:
        return

    df = df.copy()
    hue_order = [
        compressor
        for compressor in DEFAULT_COMPRESSORS
        if not df[df["compressor"] == compressor].empty
    ]

    sns.set_theme(style="whitegrid")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = [
        ("throughput_mib_s", "compression_throughput", "Throughput (MiB logical bases/s)"),
        ("p99_latency_ms", "compression_p99_latency", "p99 latency (ms)"),
        ("physical_size_gib", "compression_physical_storage", "Physical size (GiB)"),
    ]

    for metric, filename, ylabel in metrics:
        grid = sns.catplot(
            data=df,
            kind="bar",
            x="architecture",
            y=metric,
            hue="compressor",
            hue_order=hue_order,
            col="chunk_name",
            col_wrap=3,
            sharey=False,
            dodge=True,
            errorbar=None,
            height=4,
            aspect=1.25,
        )
        grid.set_axis_labels("Logical chunk size", ylabel)
        grid.set_titles("Chunk: {col_name}")
        grid.legend.set_title("Compressor")
        for axis in grid.axes.flat:
            axis.tick_params(axis="x", rotation=35)
        grid.figure.tight_layout()
        out = output_dir / "{}_{}.png".format(filename, timestamp)
        grid.figure.savefig(str(out), dpi=250, bbox_inches="tight")
        plt.close(grid.figure)
        print("Saved plot: {}".format(out), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark compression algorithms across Zarr chunk sizes and genome encodings."
    )
    parser.add_argument("--fasta", type=Path, default=Path("./hg38_data/hg38.fa"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./hg38_benchmark_data/compressed_zarr_benchmark"),
    )
    parser.add_argument("--chromosomes", nargs="+", default=["chr1"], help="Use names such as chr1 chr2, or all.")
    parser.add_argument("--architectures", nargs="+", default=list(DEFAULT_ARCHITECTURES), choices=DEFAULT_ARCHITECTURES)
    parser.add_argument("--chunk-sizes", nargs="+", default=list(DEFAULT_CHUNK_SIZES), help="Logical base chunk sizes.")
    parser.add_argument("--compressors", nargs="+", default=list(DEFAULT_COMPRESSORS), help="none zlib gzip bz2 lzma lz4 zstd")
    parser.add_argument("--num-requests", type=int, default=2000)
    parser.add_argument("--window-bases", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-page-cache",
        action="store_true",
        help="Keep the OS page cache warm during benchmark reads instead of advising chunks out.",
    )
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.build_only and args.benchmark_only:
        raise SystemExit("--build-only and --benchmark-only cannot be used together.")
    if not args.fasta.exists():
        raise SystemExit("FASTA not found: {}".format(args.fasta))
    if not args.keep_page_cache and not hasattr(os, "posix_fadvise"):
        print(
            "Warning: posix_fadvise is unavailable; benchmark reads may still hit the OS page cache.",
            flush=True,
        )

    chunk_sizes = [parse_size(label) for label in args.chunk_sizes]
    for compressor_name in args.compressors:
        validate_compressor(compressor_name)

    chromosomes = select_chromosomes(args.fasta, args.chromosomes)
    requests = generate_requests(chromosomes, args.num_requests, args.window_bases, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    print("=" * 100, flush=True)
    print("Compressed Zarr-style chunk benchmark", flush=True)
    print("FASTA: {}".format(args.fasta), flush=True)
    print("Chromosomes: {}".format(", ".join(chrom.name for chrom in chromosomes)), flush=True)
    print("Output: {}".format(args.output_dir), flush=True)
    print("=" * 100, flush=True)

    for arch in args.architectures:
        for chunk_name, chunk_bases in chunk_sizes:
            for compressor_name in args.compressors:
                path = store_path(args.output_dir, arch, chunk_name, compressor_name)
                build_seconds = 0.0

                if not args.benchmark_only:
                    print("Building {:9} chunk={:>5} compressor={:<5}".format(arch, chunk_name, compressor_name), flush=True)
                    build_seconds = build_store(
                        args.fasta,
                        path,
                        arch,
                        chunk_name,
                        chunk_bases,
                        compressor_name,
                        chromosomes,
                        args.force_rebuild,
                    )

                apparent, physical, file_count = storage_bytes(path) if path.exists() else (0, 0, 0)
                if args.build_only:
                    continue
                if not path.exists():
                    raise SystemExit("Store missing for --benchmark-only: {}".format(path))

                for iteration in range(args.iterations):
                    metrics = benchmark_store(
                        path,
                        arch,
                        chunk_bases,
                        compressor_name,
                        requests,
                        args.warmup,
                        not args.keep_page_cache,
                    )
                    row = {
                        "architecture": arch,
                        "chunk_name": chunk_name,
                        "logical_chunk_bases": chunk_bases,
                        "compressor": compressor_name,
                        "iteration": iteration + 1,
                        "num_requests": args.num_requests,
                        "window_bases": args.window_bases,
                        "build_seconds": round(build_seconds, 6),
                        "apparent_size_gib": apparent / float(1024**3),
                        "physical_size_gib": physical / float(1024**3),
                        "file_count": file_count,
                    }
                    row.update(metrics)
                    results.append(row)
                    print(
                        "{:9} | {:>5} | {:<5} | {:9.2f} MiB/s | avg {:8.3f} ms | "
                        "p99 {:8.3f} ms | physical {:.3f} GiB".format(
                            arch,
                            chunk_name,
                            compressor_name,
                            metrics["throughput_mib_s"],
                            metrics["avg_latency_ms"],
                            metrics["p99_latency_ms"],
                            row["physical_size_gib"],
                        )
                    , flush=True)

    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_csv = args.output_dir / "compression_benchmark_results_{}.csv".format(timestamp)
        write_csv(results_csv, results)
        print("\nSaved results CSV: {}".format(results_csv), flush=True)
        if not args.no_plots:
            generate_plots(results_csv, args.output_dir)
    elif args.build_only:
        print("Build-only run complete.", flush=True)


if __name__ == "__main__":
    main()
