#!/usr/bin/env python3
"""DataLoader worker/mode scaling matrix for a packed genome-zarr-4bit store.

This measures the store as a *training pipeline*, not as raw storage. For each
combination of loader mode (random map-style vs chunked iterable) and worker
count, it reports three throughputs and a GPU-starvation figure:

  * data-only   : iterate the DataLoader with no model (pure delivery rate);
  * compute-only: run the model on a synthetic in-device batch (the GPU ceiling,
                  independent of any loader);
  * end-to-end  : the real loader feeding the model.

  starvation_fraction = max(0, 1 - end_to_end / compute_only)

A value near 0 means the pipeline keeps the GPU fed at this worker count; near 1
means the GPU sits idle waiting for data. Comparing modes and worker counts
shows how many CPU workers this architecture needs to saturate the GPU, and how
much the chunked loader's decode-once amortization helps a compressed store.

It uses the package's own ``create_dataloader`` / ``create_chunked_dataloader``
and the same tiny next-base model as ``train_storage_benchmark.py``, so the
numbers reflect the shipped architecture.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from bench_models import make_backend
from train_storage_benchmark import store_metadata

try:
    from genome_zarr.dataloader import create_chunked_dataloader, create_dataloader
except ImportError:  # pragma: no cover - direct invocation without install
    from dataloader import create_chunked_dataloader, create_dataloader


FIELDNAMES = [
    "mode", "num_workers", "model", "model_params_m", "tier", "store", "compressor", "chunk_bytes",
    "window_size", "stride", "batch_size", "timed_batches", "device",
    "first_batch_seconds", "data_bases_per_second", "compute_bases_per_second",
    "e2e_bases_per_second", "starvation_fraction",
]


def make_loader(mode, store, args, num_workers):
    factory = create_chunked_dataloader if mode == "chunked" else create_dataloader
    return factory(
        store,
        args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=num_workers,
        pin_memory=(args.device == "cuda"),
        persistent_workers=num_workers > 0,
    )


def _iterate(loader, count):
    """Yield exactly ``count`` batches, restarting the loader if it runs dry."""
    produced = 0
    while produced < count:
        empty = True
        for batch in loader:
            empty = False
            yield batch
            produced += 1
            if produced >= count:
                return
        if empty:
            raise RuntimeError("DataLoader produced no batches; check window/stride vs record lengths")


def measure_data_only(loader, count, warmup):
    gen = _iterate(loader, warmup + count)
    for _ in range(warmup):
        next(gen)
    start = time.perf_counter()
    bases = 0
    for _ in range(count):
        batch = next(gen)
        bases += int(batch[:, 1:].numel())
    seconds = time.perf_counter() - start
    return bases / seconds if seconds else 0.0


def _make_backend(args, device):
    return make_backend(args.model, args.window_size, device, nt_size=args.nt_size, lr=args.learning_rate)


def measure_compute_only(args, device, count, warmup):
    import torch

    backend = _make_backend(args, device)
    backend.model.train()
    batch = backend.synthetic_batch(args.batch_size, args.window_size)

    for _ in range(warmup):
        backend.step(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    bases = 0
    for _ in range(count):
        bases += backend.step(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return (bases / seconds if seconds else 0.0), backend.describe(), backend.num_parameters


def measure_e2e(loader, args, device, count, warmup):
    import torch

    backend = _make_backend(args, device)
    backend.model.train()

    first_start = time.perf_counter()
    gen = _iterate(loader, warmup + count)
    first_batch = next(gen)
    first_batch_seconds = time.perf_counter() - first_start
    backend.step(first_batch)
    for _ in range(warmup - 1 if warmup > 0 else 0):
        backend.step(next(gen))
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    bases = 0
    for _ in range(count):
        bases += backend.step(next(gen))
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return (bases / seconds if seconds else 0.0), first_batch_seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("store")
    parser.add_argument("--window-size", type=int, default=4096)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=100, help="Timed batches per configuration")
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--workers", default="0,1,2,4,8", help="Comma-separated worker counts to sweep")
    parser.add_argument("--modes", default="random,chunked", help="Comma-separated: random, chunked")
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--model", choices=("tiny", "nt"), default="nt",
                        help="Compute backend: 'tiny' next-base conv, or 'nt' Nucleotide-Transformer-style MLM")
    parser.add_argument("--nt-size", default="50m", choices=("50m", "100m", "250m", "500m", "2b5"),
                        help="NT preset size when --model nt")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--tier", default="")
    parser.add_argument("--results", type=Path, default=Path("dataloader_scaling.csv"))
    args = parser.parse_args(argv)

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA unavailable; falling back to CPU (run on a GPU node for real numbers).", flush=True)
        args.device = "cpu"
    device = torch.device(args.device)

    worker_counts = [int(w) for w in args.workers.split(",") if w.strip() != ""]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    compressor, chunk_bytes = "", ""
    try:
        compressor, chunk_bytes = store_metadata(Path(args.store))
    except Exception as exc:  # pragma: no cover - depends on store
        print(f"Warning: could not read store metadata: {exc}", flush=True)

    if args.results.exists():
        with args.results.open("rt", newline="", encoding="utf-8") as handle:
            existing = handle.readline().rstrip("\r\n").split(",")
        if existing and existing != FIELDNAMES:
            parser.error(f"{args.results} has a different column layout; pass a new --results path")

    # Compute ceiling is loader-independent, so measure it once.
    compute_bps, model_describe, model_params = measure_compute_only(args, device, args.batches, args.warmup_batches)
    model_params_m = round(model_params / 1e6, 1)
    print(f"Model: {model_describe}", flush=True)
    print(f"Compute-only ceiling: {compute_bps:,.0f} bases/s "
          f"(device={device}, batch={args.batch_size}, window={args.window_size})\n", flush=True)

    # Write the header up front and append each row as it completes, so a crash
    # mid-sweep (e.g. an OOM worker) never loses the rows already measured.
    write_header = not args.results.exists()
    handle = args.results.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
    if write_header:
        writer.writeheader()
        handle.flush()

    count = 0
    try:
        for mode in modes:
            for num_workers in worker_counts:
                data_bps = measure_data_only(make_loader(mode, args.store, args, num_workers),
                                             args.batches, args.warmup_batches)
                e2e_bps, first_batch_seconds = measure_e2e(make_loader(mode, args.store, args, num_workers),
                                                           args, device, args.batches, args.warmup_batches)
                starvation = max(0.0, 1.0 - (e2e_bps / compute_bps)) if compute_bps else 0.0
                row = {
                    "mode": mode, "num_workers": num_workers, "model": args.model,
                    "model_params_m": model_params_m, "tier": args.tier, "store": args.store,
                    "compressor": compressor, "chunk_bytes": chunk_bytes,
                    "window_size": args.window_size, "stride": args.stride or args.window_size,
                    "batch_size": args.batch_size, "timed_batches": args.batches, "device": str(device),
                    "first_batch_seconds": first_batch_seconds, "data_bases_per_second": data_bps,
                    "compute_bases_per_second": compute_bps, "e2e_bases_per_second": e2e_bps,
                    "starvation_fraction": starvation,
                }
                writer.writerow(row)
                handle.flush()
                count += 1
                print("{mode:7s} workers={num_workers:<2d} | data {data_bases_per_second:>12,.0f} | "
                      "e2e {e2e_bases_per_second:>12,.0f} | starvation {starvation:5.1%} | "
                      "first batch {first_batch_seconds:.3f}s".format(starvation=starvation, **row), flush=True)
    finally:
        handle.close()
    print(f"\nAppended {count} rows to {args.results}")


if __name__ == "__main__":
    main()
