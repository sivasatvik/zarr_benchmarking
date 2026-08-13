# genome-zarr-4bit

`genome-zarr-4bit` turns a FASTA file into a standard, directory-backed Zarr v3
store backed by one flat packed `uint8` array. The chromosome layout is stored
in a JSON manifest at the store root, so the physical chunks are filled across
chromosome boundaries instead of creating one small array directory per record.
Each byte stores two 4-bit base codes: `A=0`, `C=1`, `G=2`, `T=3`, and `N`
(or any unsupported IUPAC base) `=4`. This is the same packed layout used by
`zarr_compression_benchmark.py`; unlike its manual backend, Zarr owns the
chunking and compression.

The implementation scans the FASTA once for record sizes and streams a single
packed array on the second pass. It does not load chromosomes into memory. The
default logical chunk is 1 MiB of bases (512 KiB packed bytes), matching the
default 1 MB base chunk.

## Install

From this directory:

```bash
python -m pip install .
```

The package requires Zarr v3 and explicitly creates every group with
`zarr_format=3`. Zstandard compression uses Zarr v3's native
`zarr.codecs.ZstdCodec`; no `numcodecs` dependency is used for this codec.
Transcode commands can read either a legacy package-created per-chromosome
store or the new flat-packed layout, but every destination they create is
Zarr v3.

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
size or Zstandard level or number of CPU workers:

```bash
# FASTA -> packed 4-bit Zarr with Zstandard compression (chunk size, level, workers)
genome-zarr fasta-to-zstd genome.fa /data/genome.zarr --chunk-bases 2097152 --zstd-level 6 --num-workers 16

# compressed -> uncompressed with parallel workers
genome-zarr decompress /data/genome.zarr /data/genome-uncompressed.zarr --num-workers 8

# uncompressed -> Zstandard (recompression) with parallel workers
genome-zarr compress /data/genome-uncompressed.zarr /data/genome-recompressed.zarr --zstd-level 6 --num-workers 8
```

`--chunk-bases` must be even, since two bases occupy each byte. Store-level
attributes document the encoding, while `manifest.json` stores each
chromosome's byte offsets and logical length. That removes the one-base padding
ambiguity for odd-length sequences and lets readers jump directly to the right
packed range without opening per-chromosome arrays.

Each command prints `Starting <command>...` immediately, then a completion
summary with elapsed time, chromosome and base counts, packed-data size,
Zarr format, compression mode, and apparent and allocated destination storage. Transcoding
commands also print the source store's apparent size.

## Python API

```python
from genome_zarr import fasta_to_zstd, decompress_zarr, compress_zarr

fasta_to_zstd("genome.fa", "genome.zarr")
decompress_zarr("genome.zarr", "genome-uncompressed.zarr")
compress_zarr("genome-uncompressed.zarr", "genome-zstd.zarr")
```

## PyTorch data loading

The PyTorch integration is optional; install it with `pip install .[torch]`
(or install a site-appropriate PyTorch build separately).

Use `GenomeZarrDataset` for fixed-length sequence windows from any store made
by `fasta-to-zstd`, `decompress`, or `compress`. It uses the manifest index,
so an odd-length record never includes its packed padding nibble. Windows do
not cross chromosomes, and Zarr is opened lazily in each DataLoader worker.
Zarr automatically decompresses Zstd chunks when they are sliced, so compressed
and uncompressed stores use the same code; do not read or decompress chunk
files yourself.

```python
from genome_zarr import create_dataloader

loader = create_dataloader(
    "/data/genome-zstd.zarr",
    window_size=4_096,
    stride=4_096,       # default: non-overlapping; use 1 for sliding windows
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

for base_codes in loader:  # torch.uint8 tensor with shape (64, 4096)
    # A=0, C=1, G=2, T=3, N=4
    train(base_codes)
```

For a large Zstd store, prefer the chunked streaming loader below. It decodes
each compressed Zarr chunk once and emits all of that chunk's windows before
moving on:

```python
from genome_zarr import create_chunked_dataloader

loader = create_chunked_dataloader(
    "genome-zarr-data/store/mm39.zarr",
    window_size=4_096,
    batch_size=64,
    shuffle=True,      # shuffles chromosome and within-chunk window order
    num_workers=4,
    pin_memory=True,
)
```

This is usually much faster for Zstd than `create_dataloader(..., shuffle=True)`.
The trade-off is deliberately local shuffling rather than a global random
permutation of every window.

If you require indexed random access (`dataset[index]`), use
`create_dataloader`. Its arrays are cached per worker, and the manifest keeps
short chromosomes cheap because the store uses one packed array instead of one
Zarr array per chromosome.

Pass `return_metadata=True` to receive a dictionary with `sequence`,
`chromosome`, `start`, and `end`; this is useful for prediction or evaluation.
For large compressed stores, keep `num_workers` modest because each worker
decompresses the chunks backing its own reads.

### Test a store

After installing the package, this command reads real batches (and therefore
tests the Zstd codec path) without running a training job:

```bash
genome-zarr-check-dataloader genome-zarr-data/store/mm39.zarr \
  --window-size 4096 --batch-size 64 --batches 20 --num-workers 4 --chunked
```

It reports the loader mode and read throughput (the non-chunked mode also
reports the declared compressor and total windows). To test one chromosome or a smaller store, add
`--chromosome chr1`; repeat `--chromosome` to select several records. Use the
same command for an output from `genome-zarr decompress`; it will report
`compressor: none`.
