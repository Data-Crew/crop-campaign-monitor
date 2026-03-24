#!/bin/bash
set -euo pipefail
STEP=$1
CONFIG=${2:-config/monitor.yaml}
shift 2 2>/dev/null || shift 1 2>/dev/null || true
OVERRIDES=("$@")
set -f  # disable glob expansion for override values containing brackets

# Resolve GPU by name → CUDA device index (fallback to GPU_ID or 0)
if [ -n "${GPU_NAME:-}" ]; then
  export CUDA_VISIBLE_DEVICES=$(python -c "from src.gpu import resolve_gpu; print(resolve_gpu('${GPU_NAME}'))")
elif [ -n "${GPU_ID:-}" ]; then
  export CUDA_VISIBLE_DEVICES=${GPU_ID}
else
  export CUDA_VISIBLE_DEVICES=0
fi

case $STEP in
  # Phase 1: Data Preparation
  ingest)   python -m src.ingest --config "$CONFIG" "${OVERRIDES[@]}" ;;
  fetch)    python -m src.fetch --config "$CONFIG" "${OVERRIDES[@]}" ;;
  chip)     python -m src.chip --config "$CONFIG" "${OVERRIDES[@]}" ;;
  # Phase 2: Training (use config/train.yaml)
  prepare)  python -m train.prepare_dataset --config "$CONFIG" "${OVERRIDES[@]}" ;;
  finetune) python -m train.finetune --config "$CONFIG" "${OVERRIDES[@]}" ;;
  export)   python -m train.export_encoder --config "$CONFIG" "${OVERRIDES[@]}" ;;
  # Phase 3: Monitor
  embed)    python -m src.embed --config "$CONFIG" "${OVERRIDES[@]}" ;;
  profile)  python -m src.profile --config "$CONFIG" "${OVERRIDES[@]}" ;;
  score)    python -m src.score --config "$CONFIG" "${OVERRIDES[@]}" ;;
  report)   python -m src.report --config "$CONFIG" "${OVERRIDES[@]}" ;;
  index)    python -m src.index --config "$CONFIG" "${OVERRIDES[@]}" ;;
  explain)  python -m src.explain --config "$CONFIG" "${OVERRIDES[@]}" ;;
  *)        echo "Unknown step: $STEP"; exit 1 ;;
esac
