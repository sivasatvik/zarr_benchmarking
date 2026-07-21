#!/bin/bash
#SBATCH --job-name=dataloader_benchmark
#SBATCH --mail-type=END,FAIL      # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sm12779@nyulangone.org
#SBATCH --partition=gpu4_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --requeue
#SBATCH --gres=gpu:1

source /gpfs/data/jt3545lab/home/sm12779/miniconda3/etc/profile.d/conda.sh
conda activate /gpfs/data/jt3545lab/home/sm12779/conda/env/test_env

# DATASET_TYPE="${1:-4bit}"
# shift || true

# python dataloader_benchmark.py --dataset-type "$DATASET_TYPE" "$@"

python dataloader_benchmark.py --dir-4bit ./zarr_compression_benchmark/stores/4bit/1MB/zstd.zarr/c/ --compressor zstd --chrom chr1 --dataset-type 4bit

python dataloader_benchmark.py --dataset-type zarr --dir-zarr ./zarr_compression_benchmark/stores/zarr/4bit/1MB/zstd.zarr --chrom chr1

python dataloader_benchmark.py --dir-4bit ./zarr_compression_benchmark/stores/4bit/1MB/lz4.zarr/c/ --compressor lz4 --chrom chr1 --dataset-type 4bit

python dataloader_benchmark.py --dataset-type zarr --dir-zarr ./zarr_compression_benchmark/stores/zarr/4bit/1MB/lz4.zarr --chrom chr1

python dataloader_benchmark.py --dir-4bit ./zarr_compression_benchmark/stores/4bit/1MB/none.zarr/c/ --compressor none --chrom chr1 --dataset-type 4bit

python dataloader_benchmark.py --dataset-type zarr --dir-zarr ./zarr_compression_benchmark/stores/zarr/4bit/1MB/none.zarr --chrom chr1