import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_file_sizes(filepath):
    """Returns (apparent_size_bytes, physical_allocated_bytes) for a file."""
    try:
        stat_info = os.stat(filepath)
        apparent = stat_info.st_size
        # st_blocks is always counted in 512-byte units on Linux
        physical = stat_info.st_blocks * 512 
        return apparent, physical
    except Exception:
        return 0, 0

def scan_storage_profiles(root_dir):
    data_points = []
    
    # 1. Scan MMAP Data
    mmap_root = os.path.join(root_dir, "mmap_data")
    if os.path.exists(mmap_root):
        for arch in ["uint8", "4bit", "2bit_flag_seq", "2bit_flag_flag"]:
            arch_dir = os.path.join(mmap_root, arch)
            if not os.path.exists(arch_dir): continue
            
            app_total, phys_total = 0, 0
            for f in os.listdir(arch_dir):
                if f.endswith(".bin"):
                    a, p = get_file_sizes(os.path.join(arch_dir, f))
                    app_total += a
                    phys_total += p
            
            # Map folder name to clean architecture name
            clean_arch = "2bit_flag" if "2bit_flag" in arch else arch
            data_points.append({
                "Architecture": clean_arch,
                "Layout": "Mmap (Flat)",
                "Apparent Size (GB)": app_total / (1024**3),
                "Physical Size (GB)": phys_total / (1024**3)
            })

    # 2. Scan Zarr V3 Data
    zarr_root = os.path.join(root_dir, "zarr_data")
    if os.path.exists(zarr_root):
        for folder in os.listdir(zarr_root):
            if not folder.endswith(".zarr"): continue
            
            # Parse architecture and chunk size from directory name
            # e.g., "2bit_flag_seq_128KB.zarr" or "uint8_1MB.zarr"
            parts = folder.replace(".zarr", "").split("_")
            chunk_name = parts[-1] # "128KB", "256KB", "512KB", "1MB", "2MB", "4MB", "16MB"
            arch_parts = parts[:-1]
            
            if "seq" in arch_parts or "flag" in arch_parts:
                clean_arch = "2bit_flag"
            else:
                clean_arch = arch_parts[0] # "uint8" or "4bit"
                
            chunk_dir = os.path.join(zarr_root, folder, "c")
            if not os.path.exists(chunk_dir): continue
            
            app_total, phys_total = 0, 0
            for f in os.listdir(chunk_dir):
                a, p = get_file_sizes(os.path.join(chunk_dir, f))
                app_total += a
                phys_total += p
                
            data_points.append({
                "Architecture": clean_arch,
                "Layout": chunk_name,
                "Apparent Size (GB)": app_total / (1024**3),
                "Physical Size (GB)": phys_total / (1024**3)
            })

    # Convert to DataFrame and aggregate (combines _seq and _flag entries for 2bit_flag)
    df = pd.DataFrame(data_points)
    df = df.groupby(["Architecture", "Layout"], as_index=False).sum()
    return df

def generate_storage_plots(df, output_dir):
    sns.set_theme(style="whitegrid")
    
    # Create a 1-row, 2-column subplot structure to compare Apparent vs Physical side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    
    # Sort order for layouts so they always group predictably
    layout_order = ["Mmap (Flat)", "128KB", "256KB", "512KB", "1MB", "2MB", "4MB", "16MB"]
    
    # Plot 1: Apparent Size (What the data should theoretically take)
    sns.barplot(
        data=df, x="Architecture", y="Apparent Size (GB)", hue="Layout", 
        hue_order=layout_order, palette="Set2", ax=axes[0]
    )
    axes[0].set_title("Apparent (Logical) Data Size\n(Theoretical Bytes Written)", fontsize=14, pad=10)
    axes[0].set_ylabel("Data Size (GB)", fontsize=12)
    axes[0].set_xlabel("Architecture", fontsize=12)
    
    # Add value labels on top of bars for clarity
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%.2f', padding=3, fontsize=9)

    # Plot 2: Physical Size (What GPFS actually allocates on the spinning disks)
    sns.barplot(
        data=df, x="Architecture", y="Physical Size (GB)", hue="Layout", 
        hue_order=layout_order, palette="tab10", ax=axes[1]
    )
    axes[1].set_title("Physical Allocated Disk Space\n(Includes GPFS Subblock Fragmentation Overhead)", fontsize=14, pad=10)
    axes[1].set_ylabel("", fontsize=12) # Shared Y-axis
    axes[1].set_xlabel("Architecture", fontsize=12)
    
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.2f', padding=3, fontsize=9)

    plt.suptitle("HG38 Genome Storage Footprint: Compression vs Filesystem Alignment", fontsize=16, y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "more_storage_architecture_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Storage profile graph successfully saved to: {output_path}")

if __name__ == "__main__":
    TARGET_DIR = "./hg38_benchmark_data"
    
    print("Scanning storage profiles... This checks actual cluster disk blocks.")
    storage_df = scan_storage_profiles(TARGET_DIR)
    
    # Print a text-based preview to the console
    print("\n--- Storage Summary (in GB) ---")
    print(storage_df.to_string(index=False, formatters={
        "Apparent Size (GB)": "{:,.2f}".format,
        "Physical Size (GB)": "{:,.2f}".format
    }))
    
    generate_storage_plots(storage_df, TARGET_DIR)