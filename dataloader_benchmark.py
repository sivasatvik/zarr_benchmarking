import os
import time
import random
import bisect
import argparse
from typing import Any
import torch
import numcodecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Sampler


class BoundedRandomSampler(Sampler):
    def __init__(self, data_source, num_samples):
        self.data_source = data_source
        self.num_samples = int(num_samples)

    def __iter__(self):
        n = len(self.data_source)
        if n <= 0 or self.num_samples <= 0:
            return iter(())
        # Sample indices lazily without creating a full randperm over massive datasets.
        random_indices = torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64)
        return iter(random_indices.tolist())

    def __len__(self):
        return self.num_samples


DATASET_TYPE_ALIASES = {
    "zarr": "zarr",
    "zarrdataset": "zarr",
    "4bit": "4bit",
    "fourbit": "4bit",
    "fourbitdataset": "4bit",
    "2bit": "2bit",
    "twobit": "2bit",
    "twobitdataset": "2bit",
}

class ZarrDataset(Dataset):
    def __init__(self, store_path, chrom, window_size=4096):
        self.store_path = store_path
        self.chrom = chrom
        self.window_size = window_size
        self.flag = False
        self.arch = "uint8"  # Zarr store is assumed to be uint8 for this benchmark
        if "4bit" in store_path.lower():
            self.arch = "4bit"
        elif "2bit" in store_path.lower():
            self.arch = "2bit"
        
        # Open once briefly just to get the total length
        import zarr
        store = zarr.open(self.store_path, mode='r')

        if self.chrom is None:
            if hasattr(store, "shape"):
                self.chrom = None
                info_array = store
            else:
                store_group: Any = store
                candidates = []
                array_keys_fn = getattr(store_group, "array_keys")
                for key in list(array_keys_fn()):
                    array = store_group[key]
                    candidates.append((key, len(array)))

                if not candidates:
                    raise ValueError(f"No arrays found in Zarr store: {self.store_path}")

                self.chrom, _ = max(candidates, key=lambda item: item[1])
                info_array = store[self.chrom]
                print(f"[*] No chromosome specified. Defaulting to largest key: '{self.chrom}'")
        else:
            info_array = store[self.chrom]

        self.total_elements = len(info_array)
        
        # Total valid windows smoothly spans the entire global length
        self.total_windows = max(0, self.total_elements - self.window_size + 1)
        
        # We leave the actual array object as None for now
        self.zarr_array = None

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        # LAZY INITIALIZATION: Open the file per-worker on the first request
        if self.zarr_array is None:
            # RULE: Force decompression to use 1 thread per PyTorch worker
            numcodecs.blosc.use_threads = False  
            
            # Open the store in read-only mode
            import zarr
            store = zarr.open(self.store_path, mode='r')
            if self.chrom is None:
                self.zarr_array = store
            else:
                self.zarr_array = store[self.chrom]

        start_byte = idx
        end_byte = idx + self.window_size
        if self.arch == "4bit":
            # For 4-bit, we need to read half the number of bytes since each byte contains 2 elements
            start_byte = idx // 2
            end_byte = (idx + self.window_size + 1) // 2  # +1 to ensure we cover the last element if odd
        # Zarr naturally handles reads across chunk boundaries here!
        window = self.zarr_array[start_byte : end_byte]

        if self.arch == "4bit":
            packed = np.asarray(window, dtype=np.uint8)
            # Unpack 4-bit values into 8-bit integers
            unpacked = np.empty(2 * len(window), dtype=np.uint8)
            unpacked[0::2] = packed >> 4
            unpacked[1::2] = packed & 0x0F
            
            # Slice to the exact window size in case of odd start index
            byte_offset = idx % 2
            window = unpacked[byte_offset : byte_offset + self.window_size]

        # Print one window for a particular index to verify correctness during debugging
        if not self.flag:
            print(f"[*] ZarrDataset: Example window {idx}: {window}")
            self.flag = True
        
        return torch.from_numpy(window).long()

class FourBitDataset(Dataset):
    def __init__(self, chunk_dir, window_size=4096, compressor="none"):
        self.window_size = window_size
        self.chunk_dir = chunk_dir
        self.compressor = compressor.lower()
        self.flag = False
        
        self.chunk_files = sorted([
            os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) 
            if os.path.isfile(os.path.join(chunk_dir, f))
        ])
        
        self.chunk_starts = []     
        self.chunk_sizes = []      
        self.max_elements_per_chunk = 0
        current_global_elements = 0
        
        if self.compressor == "zstd":
            self.decompressor = numcodecs.Zstd()

        # 1. Build Index map using actual logical chunk sizes.
        for i, f in enumerate(self.chunk_files):
            logical_bytes = self._logical_chunk_bytes(f)
            num_elements = logical_bytes * 2  # 4-bit = 2 elements per byte

            self.chunk_starts.append(current_global_elements)
            self.chunk_sizes.append(num_elements)
            self.max_elements_per_chunk = max(self.max_elements_per_chunk, num_elements)

            current_global_elements += num_elements

        self.total_elements = current_global_elements
        self.total_windows = max(0, self.total_elements - self.window_size + 1)

    def _logical_chunk_bytes(self, chunk_file):
        if self.compressor == "none":
            return os.path.getsize(chunk_file)

        with open(chunk_file, "rb") as temp_f:
            compressed_bytes = temp_f.read()
        return len(self.decompressor.decode(compressed_bytes))

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        chunk_idx = bisect.bisect_right(self.chunk_starts, idx) - 1
        local_idx = idx - self.chunk_starts[chunk_idx]
        
        elements_needed = self.window_size
        unpacked_arrays = []
        
        current_chunk_idx = chunk_idx
        current_local_idx = local_idx
        
        while elements_needed > 0:
            chunk_file = self.chunk_files[current_chunk_idx]
            chunk_total_elements = self.chunk_sizes[current_chunk_idx]
            
            elements_available = chunk_total_elements - current_local_idx
            elements_to_take = min(elements_needed, elements_available)
            
            start_byte = current_local_idx // 2
            end_byte = (current_local_idx + elements_to_take + 1) // 2 
            
            # --- COMPRESSION ROUTING ---
            if self.compressor == "zstd":
                with open(chunk_file, 'rb') as f:
                    compressed_bytes = f.read()
                raw_bytes = self.decompressor.decode(compressed_bytes)
                target_bytes = raw_bytes[start_byte:end_byte]
            else:
                with open(chunk_file, 'rb') as f:
                    f.seek(start_byte)
                    target_bytes = f.read(end_byte - start_byte)
            # ---------------------------
                
            packed = np.frombuffer(target_bytes, dtype=np.uint8)
            
            unpacked = np.empty(2 * len(packed), dtype=np.uint8)
            unpacked[0::2] = packed >> 4
            unpacked[1::2] = packed & 0x0F
            
            byte_offset = current_local_idx % 2
            window_part = unpacked[byte_offset : byte_offset + elements_to_take]
            unpacked_arrays.append(window_part)
            
            elements_needed -= elements_to_take
            current_chunk_idx += 1
            current_local_idx = 0 
            
        final_window = np.concatenate(unpacked_arrays)
        if not self.flag:
            print(f"[*] FourBitDataset: Example window {idx}: {final_window}")
            self.flag = True
        return torch.from_numpy(final_window).long()


class TwoBitFlagDataset(Dataset):
    def __init__(self, chunk_dir, window_size=4096):
        """
        Map-style dataset for 2-bit paired flag data.
        Calculates a global contiguous index across all sequence/flag chunk pairs 
        to allow windows to seamlessly span across file boundaries.
        """
        self.window_size = window_size
        self.chunk_dir = chunk_dir
        
        all_files = os.listdir(chunk_dir)
        seq_files_raw = sorted([f for f in all_files if '_seq_' in f])
        
        self.seq_files = []
        self.flag_files = []
        self.chunk_starts = []     # Global starting index of each chunk pair
        self.chunk_sizes = []      # Total valid elements in each chunk pair
        self.max_elements_per_pair = 0
        
        current_global_elements = 0
        
        # 1. Treat all valid file pairs as a single contiguous array
        for seq_file in seq_files_raw:
            flag_file = seq_file.replace('_seq_', '_flag_')
            if flag_file in all_files:
                seq_path = os.path.join(chunk_dir, seq_file)
                flag_path = os.path.join(chunk_dir, flag_file)
                
                seq_bytes = os.path.getsize(seq_path)
                flag_bytes = os.path.getsize(flag_path)
                
                seq_elements = seq_bytes * 4   # 2-bit sequence = 4 elements per byte
                flag_elements = flag_bytes * 8 # packed flags = 8 elements per byte
                
                # The safe length is constrained by whichever file is mathematically shorter
                num_elements = min(seq_elements, flag_elements)
                
                self.seq_files.append(seq_path)
                self.flag_files.append(flag_path)
                self.chunk_starts.append(current_global_elements)
                self.chunk_sizes.append(num_elements)
                
                self.max_elements_per_pair = max(self.max_elements_per_pair, num_elements)
                current_global_elements += num_elements
                
        self.total_elements = current_global_elements
        
        # Total valid windows is now calculated globally
        self.total_windows = max(0, self.total_elements - self.window_size + 1)

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        # Find which chunk pair contains the START of the window
        chunk_idx = bisect.bisect_right(self.chunk_starts, idx) - 1
        
        elements_needed = self.window_size
        seq_arrays = []
        flag_arrays = []
        
        current_chunk_idx = chunk_idx
        current_local_idx = idx - self.chunk_starts[chunk_idx]
        
        # 2. Loop dynamically in case a window spans multiple file pairs
        while elements_needed > 0:
            seq_file = self.seq_files[current_chunk_idx]
            flag_file = self.flag_files[current_chunk_idx]
            chunk_total_elements = self.chunk_sizes[current_chunk_idx]
            
            # How many elements can we safely extract from THIS specific pair?
            elements_available = chunk_total_elements - current_local_idx
            elements_to_take = min(elements_needed, elements_available)
            
            # --- Sequence Bytes Logic (4 elements per byte) ---
            seq_start_byte = current_local_idx // 4
            seq_end_byte = (current_local_idx + elements_to_take + 3) // 4
            
            with open(seq_file, 'rb') as f:
                f.seek(seq_start_byte)
                packed_seq = np.frombuffer(f.read(seq_end_byte - seq_start_byte), dtype=np.uint8)
                
            unpacked_seq = np.empty(4 * len(packed_seq), dtype=np.uint8)
            unpacked_seq[0::4] = (packed_seq >> 6) & 0x03
            unpacked_seq[1::4] = (packed_seq >> 4) & 0x03
            unpacked_seq[2::4] = (packed_seq >> 2) & 0x03
            unpacked_seq[3::4] = packed_seq & 0x03
            
            seq_offset = current_local_idx % 4
            seq_part = unpacked_seq[seq_offset : seq_offset + elements_to_take]
            seq_arrays.append(seq_part)
            
            # --- Flag Bytes Logic (8 elements per byte) ---
            flag_start_byte = current_local_idx // 8
            flag_end_byte = (current_local_idx + elements_to_take + 7) // 8
            
            with open(flag_file, 'rb') as f:
                f.seek(flag_start_byte)
                packed_flag = np.frombuffer(f.read(flag_end_byte - flag_start_byte), dtype=np.uint8)
                
            unpacked_flags = np.unpackbits(packed_flag, bitorder="big")
            
            flag_offset = current_local_idx % 8
            flag_part = unpacked_flags[flag_offset : flag_offset + elements_to_take]
            flag_arrays.append(flag_part)
            
            # --- Update counters for next file chunk (if needed) ---
            elements_needed -= elements_to_take
            current_chunk_idx += 1
            current_local_idx = 0  # Subsequent files start at their absolute beginning
            
        # 3. Stitch the sequence arrays and flag arrays together
        final_seq = np.concatenate(seq_arrays)
        final_flag = np.concatenate(flag_arrays)
        
        return {
            "sequence": torch.from_numpy(final_seq).long(),
            "flag": torch.from_numpy(final_flag).long()
        }


def benchmark_loader(dataloader, name, device, num_batches_to_test=500):
    print(f"--- Benchmarking {name} ---")
    start_time = time.perf_counter()
    latencies = []
    
    iterator = iter(dataloader)
    for i in range(num_batches_to_test):
        batch_start = time.perf_counter()
        try:
            batch = next(iterator)
            if isinstance(batch, dict):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            else:
                batch = batch.to(device, non_blocking=True)
        except StopIteration:
            break
        
        batch_time = time.perf_counter() - batch_start
        latencies.append(batch_time * 1000) # ms
        
    total_time = time.perf_counter() - start_time
    if not latencies:
        print("Result: 0.00 batches/sec | Avg Batch Latency: N/A (no batches produced)\n")
        return {"Architecture": name, "Batches_Per_Sec": 0.0, "Avg_Latency_ms": float("nan")}

    batches_per_sec = len(latencies) / total_time
    avg_latency = np.mean(latencies)
    
    print(f"Result: {batches_per_sec:.2f} batches/sec | Avg Batch Latency: {avg_latency:.2f} ms\n")
    return {"Architecture": name, "Batches_Per_Sec": batches_per_sec, "Avg_Latency_ms": avg_latency}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark zarr, 4-bit, or 2-bit dataloader performance")
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="4bit",
        help="Dataset to benchmark: zarr, 4bit, or 2bit",
    )
    parser.add_argument("--window-size", type=int, default=4096, help="Window size in elements")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=16, help="DataLoader worker count")
    parser.add_argument("--num-batches", type=int, default=5000, help="Number of batches to benchmark")
    parser.add_argument("--compressor", type=str, default="none", help="Compressor to use (none, zstd)")
    parser.add_argument(
        "--dir-zarr",
        type=str,
        default="./zarr_compression_benchmark/stores/zarr/uint8/1MB/zstd.zarr",
        help="Directory containing the Zarr store",
    )
    parser.add_argument(
        "--dir-4bit",
        type=str,
        default="./zarr_compression_benchmark/stores/4bit/1MB/c",
        help="Directory containing 4-bit chunk files",
    )
    parser.add_argument(
        "--dir-2bit",
        type=str,
        default="./zarr_compression_benchmark/stores/2bit_flag/1MB/c",
        help="Directory containing 2-bit seq/flag chunk files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataloader_benchmark",
        help="Directory to write benchmark CSV and chart",
    )
    parser.add_argument(
        "--store-path",
        type=str,
        default=None,
        help="Backward-compatible alias for --dir-zarr when benchmarking Zarr",
    )
    parser.add_argument(
        "--chrom",
        type=str,
        default=None,
        help="Chromosome key inside the Zarr store when benchmarking Zarr",
    )
    return parser.parse_args()


def normalize_dataset_type(dataset_type):
    normalized = DATASET_TYPE_ALIASES.get(dataset_type.lower())
    if normalized is None:
        valid = ", ".join(sorted({"zarr", "4bit", "2bit"}))
        raise ValueError(f"Unsupported dataset type '{dataset_type}'. Choose one of: {valid}")
    return normalized


def build_dataset(args, window_size):
    dataset_type = normalize_dataset_type(args.dataset_type)

    if dataset_type == "zarr":
        store_path = args.store_path or args.dir_zarr
        if not store_path:
            raise ValueError("Zarr benchmarking requires --dir-zarr or --store-path")
        dataset = ZarrDataset(
            store_path=store_path,
            chrom=args.chrom,
            window_size=window_size,
        )
        label = f"ZarrDataset({args.chrom or 'first-key'})"
        detail = f"[*] Using Zarr store: {store_path}"
    elif dataset_type == "4bit":
        dataset = FourBitDataset(
            args.dir_4bit,
            window_size=window_size,
            compressor=args.compressor,
        )
        label = "4bitDataset"
        detail = f"[*] Using 4-bit chunk directory: {args.dir_4bit}"
    else:
        dataset = TwoBitFlagDataset(args.dir_2bit, window_size=window_size)
        label = "2bitDataset"
        detail = f"[*] Using 2-bit chunk directory: {args.dir_2bit}"

    return dataset, label, detail

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Config
    WINDOW_SIZE = args.window_size
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    NUM_BATCHES_TO_TEST = args.num_batches
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    dataset, dataset_label, dataset_detail = build_dataset(args, WINDOW_SIZE)
    print(f"[*] Initializing {dataset_label}")
    print(dataset_detail)

    if len(dataset) == 0:
        if isinstance(dataset, FourBitDataset):
            raise ValueError(
                f"4-bit dataset produced 0 windows for window_size={WINDOW_SIZE}. "
                f"Max elements in any 4-bit chunk is {dataset.max_elements_per_chunk}. "
                "Use a smaller --window-size or point --dir-4bit to larger/uncompressed chunks."
            )
        if isinstance(dataset, TwoBitFlagDataset):
            raise ValueError(
                f"2-bit dataset produced 0 windows for window_size={WINDOW_SIZE}. "
                f"Max elements in any 2-bit seq/flag pair is {dataset.max_elements_per_pair}. "
                "Use a smaller --window-size or point --dir-2bit to larger/uncompressed chunks."
            )
        raise ValueError(
            f"Zarr dataset produced 0 windows for window_size={WINDOW_SIZE}. "
            "Use a smaller --window-size or point --dir-zarr to a longer array."
        )

    samples_to_draw = BATCH_SIZE * NUM_BATCHES_TO_TEST
    sampler = BoundedRandomSampler(dataset, samples_to_draw)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, sampler=sampler, pin_memory=True)
    
    # Warmup GPU
    if torch.cuda.is_available():
        torch.zeros(1).cuda()
    
    results = []
    results.append(benchmark_loader(dataloader, dataset_label, device, num_batches_to_test=NUM_BATCHES_TO_TEST))
    
    # --- OUTPUTS ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(results)
    csv_filename = os.path.join(output_dir, f"dataloader_benchmark_results_{timestamp}.csv")
    df.to_csv(csv_filename, index=False)
    print(f"Saved {csv_filename}")
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Architecture')
    ax1.set_ylabel('Batches per Second (Higher is Better)', color=color)
    bars = ax1.bar(df['Architecture'], df['Batches_Per_Sec'], color=color, width=0.4)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Avg Latency ms (Lower is Better)', color=color)  
    ax2.plot(df['Architecture'], df['Avg_Latency_ms'], color=color, marker='o', linestyle='dashed', linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Dataloader Performance: Zarr vs 4-bit vs 2-bit (GPU Delivery)')
    fig.tight_layout()  
    chart_filename = os.path.join(output_dir, f"dataloader_performance_chart_{timestamp}.png")
    plt.savefig(chart_filename, dpi=300)
    print(f"Saved {chart_filename}")

if __name__ == "__main__":
    main()