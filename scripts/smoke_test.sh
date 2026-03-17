#!/bin/bash
# Quick test: 5 parcels, 2 dates, small GPU, mock embeddings OK
set -euo pipefail
export GPU_NAME="RTX 500"
export SMOKE_TEST=1

echo "Smoke test — 5 parcels, GPU: ${GPU_NAME}"
bash scripts/run_data_prep.sh config/monitor.yaml
bash scripts/run_monitor.sh config/monitor.yaml
echo "Smoke test passed"
