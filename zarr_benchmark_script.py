import os
import mmap
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SECTOR_SIZE = 4096

if not hasattr(os, 'O_DIRECT'):
    raise OSError("os.O_DIRECT is missing. Run this on a Linux/HPC node.")

def read_direct_io_full(filepath: str) -> bytes:
    file_size = os.path.getsize(filepath)
    fd = os.open(filepath, os.O_RDONLY | os.O_DIRECT)
    try:
        buf = mmap.mmap(-1, file_size)
        os.readv(fd, [buf])
        return bytes(buf)
    finally:
        os.close(fd)

def read_direct_io_window(filepath: str, byte_start: int, byte_length: int) -> bytes:
    aligned_start = (byte_start // SECTOR_SIZE) * SECTOR_SIZE
    offset_in_sector = byte_start % SECTOR_SIZE
    bytes_to_read = math.ceil((offset_in_sector + byte_length) / SECTOR_SIZE) * SECTOR_SIZE
    
    fd = os.open(filepath, os.O_RDONLY | os.O_DIRECT)
    try:
        os.lseek(fd, aligned_start, os.SEEK_SET)
        buf = mmap.mmap(-1, bytes_to_read)
        os.readv(fd, [buf])
        return bytes(buf)[offset_in_sector : offset_in_sector + byte_length]
    finally:
        os.close(fd)

def run_benchmark(mode: str, arch: str, chunk_name: str, logical_chunk_size: int, requests: list, root_dir: str) -> dict:
    latencies = []
    total_bases = 0
    start_wall = time.perf_counter()
    
    for chrom, start, window_size in requests:
        req_start = time.perf_counter()
        
        if arch == "uint8":
            b_start, b_len = start, window_size
        elif arch == "4bit":
            b_start = start // 2
            b_len = ((start + window_size + 1) // 2) - b_start
        elif arch == "2bit_flag":
            seq_b_start, seq_b_len = start // 4, ((start + window_size + 3) // 4) - (start // 4)
            flg_b_start, flg_b_len = start // 8, ((start + window_size + 7) // 8) - (start // 8)

        # --- MMAP DATA ROUTING ---
        if mode == "mmap":
            mmap_dir = os.path.join(root_dir, "mmap_data")
            if arch == "uint8":
                _ = read_direct_io_window(os.path.join(mmap_dir, "uint8", f"{chrom}.bin"), b_start, b_len)
            elif arch == "4bit":
                _ = read_direct_io_window(os.path.join(mmap_dir, "4bit", f"{chrom}.bin"), b_start, b_len)
            elif arch == "2bit_flag":
                _ = read_direct_io_window(os.path.join(mmap_dir, "2bit_flag_seq", f"{chrom}.bin"), seq_b_start, seq_b_len)
                _ = read_direct_io_window(os.path.join(mmap_dir, "2bit_flag_flag", f"{chrom}.bin"), flg_b_start, flg_b_len)
                
        # --- ZARR V3 DATA ROUTING ---
        else:
            zarr_dir = os.path.join(root_dir, "zarr_data")
            chunk_idx = start // logical_chunk_size
            local_start = start % logical_chunk_size
            
            if arch == "uint8":
                c_start, c_len = local_start, window_size
            elif arch == "4bit":
                c_start, c_len = local_start // 2, ((local_start + window_size + 1) // 2) - (local_start // 2)
            elif arch == "2bit_flag":
                c_seq_start, c_seq_len = local_start // 4, ((local_start + window_size + 3) // 4) - (local_start // 4)
                c_flg_start, c_flg_len = local_start // 8, ((local_start + window_size + 7) // 8) - (local_start // 8)

            if arch == "uint8":
                path = os.path.join(zarr_dir, f"uint8_{chunk_name}.zarr", "c", f"{chrom}_{chunk_idx}")
                _ = read_direct_io_full(path)[c_start : c_start + c_len]
            elif arch == "4bit":
                path = os.path.join(zarr_dir, f"4bit_{chunk_name}.zarr", "c", f"{chrom}_{chunk_idx}")
                _ = read_direct_io_full(path)[c_start : c_start + c_len]
            elif arch == "2bit_flag":
                seq_path = os.path.join(zarr_dir, f"2bit_flag_seq_{chunk_name}.zarr", "c", f"{chrom}_{chunk_idx}")
                flg_path = os.path.join(zarr_dir, f"2bit_flag_flag_{chunk_name}.zarr", "c", f"{chrom}_{chunk_idx}")
                _ = read_direct_io_full(seq_path)[c_seq_start : c_seq_start + c_seq_len]
                _ = read_direct_io_full(flg_path)[c_flg_start : c_flg_start + c_flg_len]

        latencies.append(time.perf_counter() - req_start)
        total_bases += window_size

    wall_time = time.perf_counter() - start_wall
    throughput_mb_s = (total_bases / (1024 * 1024)) / wall_time
    latencies_ms = np.array(latencies) * 1000
    
    display_chunk = "Mmap (Flat)" if mode == "mmap" else chunk_name
    print(f"{arch:<12} | {display_chunk:<11} | {throughput_mb_s:>10.2f} MB/s | {np.mean(latencies_ms):>11.2f} ms | {np.percentile(latencies_ms, 99):>11.2f} ms")
    
    # Return the metrics for graphing
    return {
        "Architecture": arch,
        "Layout": display_chunk,
        "Throughput (MB/s)": throughput_mb_s,
        "Avg Latency (ms)": np.mean(latencies_ms),
        "p99 Latency (ms)": np.percentile(latencies_ms, 99)
    }

def generate_graphs(results: list, output_dir: str):
    """Generates and saves bar plots from the benchmark results with individual run points."""
    print("\nGenerating graphs...")
    df = pd.DataFrame(results)
    
    # Set plot style
    sns.set_theme(style="whitegrid")
    
    # 1. Throughput Graph
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Architecture", y="Throughput (MB/s)", hue="Layout", palette="Set2", errorbar=None)
    sns.stripplot(data=df, x="Architecture", y="Throughput (MB/s)", hue="Layout", palette="Set2", 
                  dodge=True, alpha=0.6, size=5, legend=False)
    plt.title("GPFS Direct I/O Benchmark: Logical Throughput", fontsize=14, pad=15)
    plt.ylabel("Throughput (MB/s)", fontsize=12)
    plt.xlabel("Encoding Architecture", fontsize=12)
    plt.legend(title="Data Layout", loc="upper right")
    plt.tight_layout()
    throughput_path = os.path.join(output_dir, "benchmark_throughput_multi.png")
    plt.savefig(throughput_path, dpi=300)
    plt.close()
    
    # 2. Latency Graph (p99)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Architecture", y="p99 Latency (ms)", hue="Layout", palette="tab10", errorbar=None)
    sns.stripplot(data=df, x="Architecture", y="p99 Latency (ms)", hue="Layout", palette="tab10", 
                  dodge=True, alpha=0.6, size=5, legend=False)
    plt.title("GPFS Direct I/O Benchmark: 99th Percentile Latency", fontsize=14, pad=15)
    plt.ylabel("Latency (ms) - Lower is better", fontsize=12)
    plt.xlabel("Encoding Architecture", fontsize=12)
    plt.legend(title="Data Layout", loc="upper right")
    plt.tight_layout()
    latency_path = os.path.join(output_dir, "benchmark_latency_multi.png")
    plt.savefig(latency_path, dpi=300)
    plt.close()
    
    print(f"Saved: {throughput_path}")
    print(f"Saved: {latency_path}")


if __name__ == "__main__":
    ROOT_DIR = "./hg38_benchmark_data"
    NUM_REQUESTS = 1000
    WINDOW_SIZE = 4096
    
    CHUNK_SIZES = {
        "128KB": 131072,
        "256KB": 262144,
        "512KB": 524288,
        "1MB": 1048576,
        "2MB": 2097152,
        "4MB": 4194304,
        "16MB": 16777216
    }
    ARCHITECTURES = ["uint8", "4bit", "2bit_flag"]
    
    CHROM_LENGTH = 248956422 # chr1 bounds for testing
    np.random.seed(42)
    starts = np.random.randint(0, CHROM_LENGTH - WINDOW_SIZE, size=NUM_REQUESTS)
    requests = [("chr1", start, WINDOW_SIZE) for start in starts]

    print("=" * 72)
    print("             GPFS DIRECT I/O BENCHMARK RESULTS (SSD)")
    print("             Running 3 iterations per configuration")
    print("=" * 72)
    print(f"{'Architecture':<12} | {'Data Layout':<11} | {'Logical T/P':>10} | {'Avg Latency':>14} | {'p99 Latency':>14}")
    print("-" * 72)
    
    all_results = []
    NUM_ITERATIONS = 3
    
    # Run benchmarks 3 times for each architecture with mmap
    for iteration in range(NUM_ITERATIONS):
        for arch in ARCHITECTURES:
            res = run_benchmark("mmap", arch, "Mmap", None, requests, ROOT_DIR)
            all_results.append(res)
        
    print("-" * 72)
    
    # Run benchmarks 3 times for each architecture/chunk combination with zarr
    for iteration in range(NUM_ITERATIONS):
        for arch in ARCHITECTURES:
            for chunk_name, size in CHUNK_SIZES.items():
                res = run_benchmark("zarr", arch, chunk_name, size, requests, ROOT_DIR)
                all_results.append(res)
            
    print("=" * 72)
    
    # Trigger the graphing function
    generate_graphs(all_results, ROOT_DIR)