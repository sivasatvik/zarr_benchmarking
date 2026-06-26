import os
import json
import math
import numpy as np
from pyfaidx import Fasta
from tqdm import tqdm

CHUNK_SIZES = {
    "128KB": 131072,
    "256KB": 262144,
    "512KB": 524288,
    "1MB": 1048576,
    "2MB": 2097152,
    "4MB": 4194304,
    "16MB": 16777216
}

def pad_for_direct_io(data: bytes) -> bytes:
    """Pads byte arrays to a multiple of 4096 for OS Direct I/O."""
    remainder = len(data) % 4096
    return data + (b'\x00' * (4096 - remainder)) if remainder else data

class Codecs:
    # (Same encoding logic as before)
    @staticmethod
    def encode_uint8(seq: str) -> bytes:
        char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        return np.array([char_map.get(c, 4) for c in seq.upper()], dtype=np.uint8).tobytes()

    @staticmethod
    def encode_4bit(seq: str) -> bytes:
        base_bits = {'A': 0b00, 'C': 0b01, 'G': 0b10, 'T': 0b11}
        byte_arr = bytearray()
        length = len(seq)
        for i in range(0, length, 2):
            b1 = seq[i].upper()
            n1 = 0b01 if b1 == 'N' else (base_bits.get(b1, 0b00) << 2)
            if i + 1 < length:
                b2 = seq[i+1].upper()
                n2 = 0b01 if b2 == 'N' else (base_bits.get(b2, 0b00) << 2)
            else:
                n2 = 0b00
            byte_arr.append((n1 << 4) | n2)
        return bytes(byte_arr)

    @staticmethod
    def encode_2bit_flag(seq: str) -> tuple:
        encode_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
        length = len(seq)
        pad_seq = seq + ('A' * ((4 - (length % 4)) % 4))
        seq_bytes = bytearray()
        for i in range(0, len(pad_seq), 4):
            val = 0
            for pos in range(4): val |= (encode_map.get(pad_seq[i+pos].upper(), 0) << (6 - 2*pos))
            seq_bytes.append(val)
            
        pad_flag = seq + ('A' * ((8 - (length % 8)) % 8))
        flag_bytes = bytearray()
        for i in range(0, len(pad_flag), 8):
            val = 0
            for pos in range(8): val |= ((1 if pad_flag[i+pos].upper() == 'N' else 0) << (7 - pos))
            flag_bytes.append(val)
            
        return bytes(seq_bytes), bytes(flag_bytes)

def write_zarr_v3_meta(path: str, dtype: str, chunk_shape: list):
    """Writes a minimal valid Zarr V3 array metadata file."""
    meta = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [0], # Placeholder for concatenation
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": chunk_shape}},
        "data_type": dtype,
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "_"}}
    }
    with open(os.path.join(path, "zarr.json"), "w") as f:
        json.dump(meta, f, indent=2)

def build_datasets(fasta_path: str, root_out_dir: str):
    genome = Fasta(fasta_path)
    chromosomes = [c for c in genome.keys() if c.startswith('chr')]

    mmap_dir = os.path.join(root_out_dir, "mmap_data")
    zarr_dir = os.path.join(root_out_dir, "zarr_data")
    arch_names = ["uint8", "4bit", "2bit_flag_seq", "2bit_flag_flag"]
    generated_mmap_files = 0
    generated_zarr_chunks = 0

    # MMAP: generate only missing chromosome files for each architecture.
    for arch in arch_names:
        arch_dir = os.path.join(mmap_dir, arch)
        os.makedirs(arch_dir, exist_ok=True)
        for chrom in chromosomes:
            out_file = os.path.join(arch_dir, f"{chrom}.bin")
            if os.path.exists(out_file):
                continue

            chrom_seq = str(genome[chrom][:].seq)

            if arch == "uint8":
                data = Codecs.encode_uint8(chrom_seq)
            elif arch == "4bit":
                data = Codecs.encode_4bit(chrom_seq)
            else:
                seq_f, flag_f = Codecs.encode_2bit_flag(chrom_seq)
                data = seq_f if arch == "2bit_flag_seq" else flag_f

            with open(out_file, "wb") as f:
                f.write(pad_for_direct_io(data))
            generated_mmap_files += 1

    # Zarr: generate only missing chromosome chunk files for each architecture/chunk size.
    for chunk_name, chunk_size in CHUNK_SIZES.items():
        for arch in arch_names:
            array_dir = os.path.join(zarr_dir, f"{arch}_{chunk_name}.zarr")
            chunk_dir = os.path.join(array_dir, "c")
            os.makedirs(chunk_dir, exist_ok=True)

            meta_path = os.path.join(array_dir, "zarr.json")
            if not os.path.exists(meta_path):
                write_zarr_v3_meta(array_dir, "uint8", [chunk_size])

            for chrom in chromosomes:
                chrom_seq = str(genome[chrom][:].seq)
                chrom_len = len(chrom_seq)
                num_chunks = math.ceil(chrom_len / chunk_size)
                missing_indices = [
                    i for i in range(num_chunks)
                    if not os.path.exists(os.path.join(chunk_dir, f"{chrom}_{i}"))
                ]

                if not missing_indices:
                    continue

                for i in tqdm(missing_indices, leave=False, desc=f"Zarr {chunk_name} {chrom}"):
                    start = i * chunk_size
                    chunk_str = chrom_seq[start : min((i + 1) * chunk_size, chrom_len)]

                    if arch == "uint8":
                        data = Codecs.encode_uint8(chunk_str)
                    elif arch == "4bit":
                        data = Codecs.encode_4bit(chunk_str)
                    else:
                        seq_b, flag_b = Codecs.encode_2bit_flag(chunk_str)
                        data = seq_b if arch == "2bit_flag_seq" else flag_b

                    with open(os.path.join(chunk_dir, f"{chrom}_{i}"), "wb") as f:
                        f.write(pad_for_direct_io(data))
                    generated_zarr_chunks += 1

    if generated_mmap_files == 0 and generated_zarr_chunks == 0:
        print("All chromosome files already exist; nothing new was generated.")
    else:
        print(
            f"Generated {generated_mmap_files} mmap files and "
            f"{generated_zarr_chunks} zarr chunk files."
        )

if __name__ == "__main__":
    build_datasets("./hg38_data/hg38.fa", "./hg38_benchmark_data")