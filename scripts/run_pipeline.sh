#!/bin/bash
# Full end-to-end pipeline: Data Prep → Training → Monitor
# For individual phases, use:
#   bash scripts/run_data_prep.sh   (Phase 1)
#   bash scripts/run_training.sh    (Phase 2)
#   bash scripts/run_monitor.sh     (Phase 3)
set -euo pipefail

MONITOR_CONFIG=${1:-config/monitor.yaml}
TRAIN_CONFIG=${2:-config/train.yaml}

echo "╔══════════════════════════════════════════╗"
echo "║   Full Pipeline — All Phases             ║"
echo "╚══════════════════════════════════════════╝"
echo ""

bash scripts/run_data_prep.sh $MONITOR_CONFIG
echo ""
bash scripts/run_training.sh $TRAIN_CONFIG
echo ""
bash scripts/run_monitor.sh $MONITOR_CONFIG

echo ""
echo "Full pipeline complete."
