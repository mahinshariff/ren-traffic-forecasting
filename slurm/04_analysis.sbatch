#!/bin/bash
#SBATCH --job-name=renfc-analysis
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# Runs after the sweep: sweep report, distribution, commonality, figures.
mkdir -p slurm_logs
source slurm/common.sh
python - <<'EOF'
import os
from renfc.config import load_config
from renfc.coverage import load_selection
from renfc.experiment import write_report
cfg = load_config(os.environ["CONFIG"])
write_report(cfg, load_selection(cfg)["routers"])
EOF
python -m renfc distribution -c "$CONFIG"
python -m renfc commonality -c "$CONFIG"
python -m renfc figures -c "$CONFIG"
