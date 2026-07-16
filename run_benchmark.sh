#!/bin/bash
#SBATCH --job-name=zarr_benchmark
#SBATCH --mail-type=END,FAIL      # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sm12779@nyulangone.org
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --requeue

source /gpfs/data/jt3545lab/home/sm12779/miniconda3/etc/profile.d/conda.sh
conda activate /gpfs/data/jt3545lab/home/sm12779/conda/env/test_env

python zarr_benchmark_script.py
