#!/bin/bash
# Sourced by every SLURM script: activates the environment and sets paths.
module purge
module load anaconda 2>/dev/null || module load mamba 2>/dev/null || true
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-renfc}"

export CONFIG=${CONFIG:-configs/default.yaml}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "job ${SLURM_JOB_ID:-local} on $(hostname) | config=${CONFIG} | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"
