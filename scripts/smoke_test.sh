#!/bin/bash
# End-to-end test of the pipeline on synthetic data (CPU, a few minutes).
#   bash scripts/smoke_test.sh [work_dir]
set -euo pipefail
WORK=${1:-/tmp/renfc_smoke}
export RENFC_RAW_DIR="$WORK/raw" RENFC_OUT_DIR="$WORK/out"
rm -rf "$WORK"
python scripts/make_synthetic_raw.py --out "$RENFC_RAW_DIR" --days 21
python -m renfc all -c configs/smoke.yaml
echo
echo "Reports:"; ls "$RENFC_OUT_DIR/reports"
echo "Figures:"; ls "$RENFC_OUT_DIR/figures" | grep pdf
echo "Tables:";  ls "$RENFC_OUT_DIR/tables"
echo "SMOKE TEST PASSED"
