#!/bin/bash
#SBATCH --job-name=renfc-sweep
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-3
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
# One array task per selected router (index into coverage/selection.json).
# Re-submitting after a time-out resumes: finished runs are skipped.
mkdir -p slurm_logs
source slurm/common.sh
python -m renfc sweep -c "$CONFIG" --router-index "${SLURM_ARRAY_TASK_ID}" --no-report
