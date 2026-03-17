#!/bin/bash
set -euo pipefail

CONFIG=${1:-config/monitor.yaml}
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
dev = cfg.get('gpu', {}).get('default_device', '0')
print(resolve_gpu(dev))
" 2>/dev/null || echo "0")
fi

echo "╔══════════════════════════════════════════╗"
echo "║   Phase 1 — Data Preparation             ║"
echo "╚══════════════════════════════════════════╝"
echo "Config: $CONFIG"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

echo "[1/3] Ingesting parcels + crop labels..."
python -m src.ingest --config "$CONFIG" "${OVERRIDES[@]}"

echo "[2/3] Fetching Sentinel-2 imagery..."
python -m src.fetch --config "$CONFIG" "${OVERRIDES[@]}"

echo "[3/3] Extracting chips..."
python -m src.chip --config "$CONFIG" "${OVERRIDES[@]}"

echo ""
echo "Data preparation complete."
echo "   Chips in: data/chips/"
echo "   Next: run training (scripts/run_training.sh) or monitor (scripts/run_monitor.sh)"
