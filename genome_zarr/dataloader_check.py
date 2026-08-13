"""Read a bounded number of batches to validate a genome Zarr data loader."""

from __future__ import annotations

import argparse
import time

try:  # Supports both installed-module and ``python genome_zarr/...py`` use.
    from .dataloader import create_chunked_dataloader, create_dataloader
except ImportError:  # pragma: no cover - direct script invocation
    from dataloader import create_chunked_dataloader, create_dataloader


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="genome-zarr-check-dataloader",
        description="Read batches from a packed genome-zarr-4bit store.",
    )
    parser.add_argument("store", help="Path to a store made by genome-zarr")
    parser.add_argument("--window-size", type=int, default=4096)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=10, help="Number of batches to read")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chromosome", action="append", dest="chromosomes", help="Limit to one or more record names")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--chunked", action="store_true", help="Stream chunk-local windows (recommended for Zstd stores)")
    args = parser.parse_args(argv)
    if args.batches <= 0:
        parser.error("--batches must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")

    factory = create_chunked_dataloader if args.chunked else create_dataloader
    loader = factory(
        args.store,
        args.window_size,
        stride=args.stride,
        chromosomes=args.chromosomes,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        persistent_workers=False,
    )
    dataset = loader.dataset
    if args.chunked:
        print("Store: {} | loader: chunked streaming".format(args.store))
    else:
        print(
            "Store: {} | compressor: {} | records: {} | windows: {}".format(
                args.store, dataset.compressor, len(dataset.chromosomes), len(dataset)
            )
        )
    started = time.perf_counter()
    bases = 0
    read_batches = 0
    for batch in loader:
        bases += int(batch.numel())
        read_batches += 1
        if read_batches == args.batches:
            break
    elapsed = time.perf_counter() - started
    rate = bases / elapsed if elapsed else 0.0
    print("Read {} batches, {:,} bases in {:.3f}s ({:,.0f} bases/s)".format(read_batches, bases, elapsed, rate))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
