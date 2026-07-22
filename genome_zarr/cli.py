"""Command-line interface for genome-zarr-4bit."""

import argparse

from .convert import compress_zarr, decompress_zarr, fasta_to_zstd


def _add_overwrite(parser):
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing destination store.")


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
        if args.command == "fasta-to-zstd":
            output = fasta_to_zstd(args.fasta, args.destination, chunk_bases=args.chunk_bases, zstd_level=args.zstd_level, overwrite=args.overwrite)
        elif args.command == "decompress":
            output = decompress_zarr(args.source, args.destination, overwrite=args.overwrite)
        else:
            output = compress_zarr(args.source, args.destination, zstd_level=args.zstd_level, overwrite=args.overwrite)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
