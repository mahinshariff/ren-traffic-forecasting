#!/bin/bash
# Collect the state of a run into ONE text file: stage reports, the window
# selection, SLURM queue state, the tail of every log and an inventory of the
# output directory. Convenient for reviewing a cluster run from elsewhere.
#
#   bash scripts/collect_run_summary.sh [config.yaml]
set -uo pipefail
CONFIG=${1:-configs/default.yaml}
OUT_DIR=$(python - "$CONFIG" <<'EOF'
import sys
from renfc.config import load_config
print(load_config(sys.argv[1]).paths.out_dir)
EOF
)
STAMP=$(date +%Y%m%d_%H%M%S)
FILE="run_summary_${STAMP}.txt"
{
  echo "===== renfc run summary ${STAMP} ====="
  echo "host: $(hostname)   user: ${USER:-?}   pwd: $(pwd)"
  echo "config: ${CONFIG}   out_dir: ${OUT_DIR}"
  echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a') $(git status --porcelain 2>/dev/null | wc -l | xargs) modified files"
  echo
  echo "===== squeue (my jobs) ====="
  squeue -u "${USER:-}" -o "%.10i %.20j %.8T %.10M %.6D %R" 2>/dev/null || echo "squeue unavailable"
  echo
  echo "===== recent sacct ====="
  sacct -u "${USER:-}" --starttime "$(date -d '-3 days' +%F 2>/dev/null || date -v-3d +%F)" \
    --format=JobID%14,JobName%22,State%12,Elapsed%10,MaxRSS%10,ExitCode 2>/dev/null | head -40 || true
  echo
  echo "===== selection.json ====="
  cat "${OUT_DIR}/coverage/selection.json" 2>/dev/null || echo "(coverage stage not run yet)"
  echo
  for f in "${OUT_DIR}"/reports/*.md; do
    [ -f "$f" ] || continue
    echo; echo "===== REPORT: $(basename "$f") ====="; cat "$f"
  done
  echo
  echo "===== results summary ====="
  if [ -f "${OUT_DIR}/results/sweep_results.csv" ]; then
    python - "${OUT_DIR}/results/sweep_results.csv" <<'EOF'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
print("rows", len(df), "runs", df.run_id.nunique())
print(df.groupby(["router"]).run_id.nunique().to_string())
print(df.groupby("model")[["sMAPE", "R2"]].median().round(3).to_string())
EOF
    echo "done markers: $(ls "${OUT_DIR}/results/done" 2>/dev/null | wc -l | xargs)"
  else
    echo "(no sweep results yet)"
  fi
  echo
  for f in "${OUT_DIR}"/logs/*.log slurm_logs/*.out slurm_logs/*.err; do
    [ -f "$f" ] || continue
    echo; echo "===== LOG TAIL: $f ====="; tail -n 40 "$f"
  done
  echo
  echo "===== output inventory ====="
  find "${OUT_DIR}" -maxdepth 2 -type d -exec sh -c 'printf "%6s  %s\n" "$(find "$1" -type f | wc -l | xargs)" "$1"' _ {} \; 2>/dev/null | sort -k2
} > "${FILE}" 2>&1
echo "wrote ${FILE} ($(wc -l < "${FILE}") lines)"
