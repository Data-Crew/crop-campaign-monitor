#!/bin/bash
set -euo pipefail

CONFIG=${1:-config/train.yaml}
shift 2>/dev/null || true
OVERRIDES=("$@")
set -f  # disable glob expansion for override values containing brackets

# Resolve GPU by name (from GPU_NAME env or config) → CUDA device index
if [ -n "${GPU_NAME:-}" ]; then
  export CUDA_VISIBLE_DEVICES=$(python -c "from src.gpu import resolve_gpu; print(resolve_gpu('${GPU_NAME}'))")
elif [ -n "${GPU_ID:-}" ]; then
  export CUDA_VISIBLE_DEVICES=${GPU_ID}
else
  export CUDA_VISIBLE_DEVICES=$(python -c "
from src.gpu import resolve_gpu
from src.config import get_config
cfg = get_config('${CONFIG}', resolve_paths=False)
dev = cfg.get('gpu', {}).get('device', '0')
print(resolve_gpu(dev))
" 2>/dev/null || echo "0")
fi

echo "╔══════════════════════════════════════════╗"
echo "║   Phase 2 — Training                     ║"
echo "╚══════════════════════════════════════════╝"
echo "Config: $CONFIG"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Pre-flight checks
if [ ! -f "data/output/parcels_labeled.parquet" ]; then
  echo "ERROR: No labeled parcels found (data/output/parcels_labeled.parquet)."
  echo "       Run Phase 1 first: bash scripts/run_data_prep.sh"
  exit 1
fi

if [ ! -d "data/chips" ] || [ -z "$(ls -A data/chips 2>/dev/null)" ]; then
  echo "WARNING: No chips found in data/chips/."
  echo "         Training will fail at prepare_dataset unless chips exist."
  echo "         Run Phase 1 first: bash scripts/run_data_prep.sh"
  echo ""
fi

echo "[1/3] Preparing training dataset..."
python -m train.prepare_dataset --config "$CONFIG" "${OVERRIDES[@]}"

echo "[2/3] Fine-tuning Clay backbone..."
python -m train.finetune --config "$CONFIG" "${OVERRIDES[@]}"

echo "[3/3] Exporting encoder for inference..."
python -m train.export_encoder --config "$CONFIG" "${OVERRIDES[@]}"

echo ""
echo "Training complete."
echo "   Encoder saved to: data/model/clay-crop-encoder.ckpt"
echo "   Next: run the monitor phase: bash scripts/run_monitor.sh"
