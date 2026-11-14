#!/bin/bash
# One-time environment setup on the cluster (run on a login node).
#
#   bash slurm/env_setup.sh
#
# Creates a conda environment named "renfc" with a CUDA-enabled PyTorch and
# installs this package in editable mode. Adjust the module name if your
# cluster exposes conda differently (e.g. `module load mamba`).
set -euo pipefail

module purge
module load anaconda 2>/dev/null || module load mamba 2>/dev/null || true

ENV_NAME=${ENV_NAME:-renfc}
if conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "conda env '${ENV_NAME}' already exists; updating packages"
else
  conda create -y -n "${ENV_NAME}" python=3.11
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# CUDA build of PyTorch from the official index (CPU-only fallback: drop --index-url).
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121 || pip install torch
pip install -e ".[dev]"

python - <<'EOF'
import torch, renfc
print("renfc", renfc.__version__, "| torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
EOF
echo "Environment '${ENV_NAME}' ready."
