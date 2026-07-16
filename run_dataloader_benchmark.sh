#!/bin/bash
#SBATCH --job-name=streaming_dataset_benchmark
#SBATCH --mail-type=END,FAIL      # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sm12779@nyulangone.org
#SBATCH --partition=gpu4_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --requeue
#SBATCH --gres=gpu:4

source /gpfs/data/jt3545lab/home/sm12779/miniconda3/etc/profile.d/conda.sh
conda activate /gpfs/data/jt3545lab/home/sm12779/conda/env/test_env

DATASET_TYPE="${1:-4bit}"
shift || true

python dataloader_benchmark.py --dataset-type "$DATASET_TYPE" "$@"
