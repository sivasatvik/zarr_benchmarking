"""Command-line interface for genome-zarr-4bit."""

import argparse
import time

from .convert import compress_zarr, decompress_zarr, fasta_to_zstd, store_statistics


def _add_overwrite(parser):
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing destination store.")


def _format_bytes(value):
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return "{:.2f} {}".format(amount, unit)
        amount /= 1024


def _format_duration(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02d}:{:02d}:{:05.2f}".format(int(hours), int(minutes), seconds)


def _print_completion(command, source, output, elapsed, source_stats=None):
    stats = store_statistics(output)
    print("Completed: {}".format(command))
    if source is not None:
        print("Source: {}".format(source))
    print("Destination: {}".format(output))
    print("Elapsed: {} ({:.3f} seconds)".format(_format_duration(elapsed), elapsed))
    print("Chromosomes: {:,}".format(stats["chromosomes"]))
    print("Logical bases: {:,}".format(stats["logical_bases"]))
    print("Packed 4-bit data: {}".format(_format_bytes(stats["packed_bytes"])))
    print("Zarr format: {}".format(stats["zarr_format"]))
    print("Compression: {}".format(stats["compressor"]))
    print("Destination apparent size: {}".format(_format_bytes(stats["apparent_bytes"])))
    print("Destination allocated size: {}".format(_format_bytes(stats["allocated_bytes"])))
    if source_stats is not None:
        print("Source Zarr format: {}".format(source_stats["zarr_format"]))
        print("Source apparent size: {}".format(_format_bytes(source_stats["apparent_bytes"])))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="genome-zarr", description="Convert FASTA and packed 4-bit Zarr genome stores.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fasta = subparsers.add_parser("fasta-to-zstd", help="Create a Zstd-compressed 4-bit store from FASTA.")
    fasta.add_argument("fasta")
    fasta.add_argument("destination")
    fasta.add_argument("--chunk-bases", type=int, default=1_048_576, help="Even number of bases per Zarr chunk (default: 1048576).")
    fasta.add_argument("--zstd-level", type=int, default=3)
    _add_overwrite(fasta)

    decompress = subparsers.add_parser("decompress", help="Create an uncompressed copy of a packed store.")
    decompress.add_argument("source")
    decompress.add_argument("destination")
    _add_overwrite(decompress)

    compress = subparsers.add_parser("compress", help="Create a Zstd-compressed copy of a packed store.")
    compress.add_argument("source")
    compress.add_argument("destination")
    compress.add_argument("--zstd-level", type=int, default=3)
    _add_overwrite(compress)

    args = parser.parse_args(argv)
    try:
        source = getattr(args, "source", None)
        source_stats = store_statistics(source) if source is not None else None
        print("Starting {}...".format(args.command), flush=True)
        started = time.perf_counter()
        if args.command == "fasta-to-zstd":
            output = fasta_to_zstd(args.fasta, args.destination, chunk_bases=args.chunk_bases, zstd_level=args.zstd_level, overwrite=args.overwrite)
        elif args.command == "decompress":
            output = decompress_zarr(args.source, args.destination, overwrite=args.overwrite)
        else:
            output = compress_zarr(args.source, args.destination, zstd_level=args.zstd_level, overwrite=args.overwrite)
        elapsed = time.perf_counter() - started
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    _print_completion(args.command, source, output, elapsed, source_stats)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
