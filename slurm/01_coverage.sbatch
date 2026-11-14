#!/bin/bash
#SBATCH --job-name=renfc-coverage
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
mkdir -p slurm_logs
source slurm/common.sh
python -m renfc coverage -c "$CONFIG"
