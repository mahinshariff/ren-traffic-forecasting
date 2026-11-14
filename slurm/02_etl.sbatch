#!/bin/bash
#SBATCH --job-name=renfc-etl
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# Optional: restrict to some routers, e.g.  sbatch --export=ALL,ROUTERS="batonrouge" slurm/02_etl.sbatch
mkdir -p slurm_logs
source slurm/common.sh
python -m renfc etl -c "$CONFIG" ${ROUTERS:+--routers $ROUTERS}
