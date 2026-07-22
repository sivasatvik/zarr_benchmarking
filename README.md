# genome-zarr-4bit

`genome-zarr-4bit` turns a FASTA file into a standard, directory-backed Zarr v2
group. Each chromosome is a one-dimensional `uint8` Zarr array containing two
4-bit base codes per byte: `A=0`, `C=1`, `G=2`, `T=3`, and `N` (or any
unsupported IUPAC base) `=4`. This is the same packed layout used by
`zarr_compression_benchmark.py`; unlike its manual backend, Zarr and
`numcodecs` own the chunking and compression.

The implementation scans the FASTA once for record sizes and streams it on the
second pass. It does not load chromosomes into memory. The default logical
chunk is 1 MiB of bases (512 KiB packed bytes), matching the benchmark's
default 1 MB base chunk.

## Install

From this directory:

```bash
python -m pip install .
```

This deliberately uses the stable Zarr v2 API because its `numcodecs.Zstd`
compressor is the format used in the existing benchmark. The package pins
`zarr<3` to make stores reproducible across installations.

## Commands

```bash
# FASTA -> packed 4-bit Zarr with Zstandard compression
genome-zarr fasta-to-zstd genome.fa /data/genome.zarr

# compressed -> uncompressed, preserving arrays, chunks, and attributes
genome-zarr decompress /data/genome.zarr /data/genome-uncompressed.zarr

# uncompressed -> Zstandard (also works as a recompression operation)
genome-zarr compress /data/genome-uncompressed.zarr /data/genome-recompressed.zarr
```

Destinations must be new unless `--overwrite` is supplied. To select a chunk
size or Zstandard level:

```bash
genome-zarr fasta-to-zstd genome.fa /data/genome.zarr --chunk-bases 2097152 --zstd-level 6
```

`--chunk-bases` must be even, since two bases occupy each byte. Store-level
attributes document the encoding and each chromosome array has a
`logical_length` attribute, which removes the one-base padding ambiguity for
odd-length sequences.

Each command prints `Starting <command>...` immediately, then a completion
summary with elapsed time, chromosome and base counts, packed-data size,
compression mode, and apparent and allocated destination storage. Transcoding
commands also print the source store's apparent size.

## Python API

```python
from genome_zarr import fasta_to_zstd, decompress_zarr, compress_zarr

fasta_to_zstd("genome.fa", "genome.zarr")
decompress_zarr("genome.zarr", "genome-uncompressed.zarr")
compress_zarr("genome-uncompressed.zarr", "genome-zstd.zarr")
```
