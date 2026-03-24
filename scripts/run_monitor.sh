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
echo "║   Phase 3 — Monitor                      ║"
echo "╚══════════════════════════════════════════╝"
echo "Config: $CONFIG"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Pre-flight check
if [ ! -d "data/chips" ] || [ -z "$(ls -A data/chips 2>/dev/null)" ]; then
  echo "ERROR: No chips found in data/chips/."
  echo "       Run Phase 1 first: bash scripts/run_data_prep.sh"
  exit 1
fi

echo "[1/5] Generating embeddings (GPU)..."
python -m src.embed --config "$CONFIG" "${OVERRIDES[@]}"

echo "[2/5] Building reference profiles..."
python -m src.profile --config "$CONFIG" "${OVERRIDES[@]}"

echo "[3/5] Scoring parcels..."
python -m src.score --config "$CONFIG" "${OVERRIDES[@]}"

echo "[4/5] Generating report..."
python -m src.report --config "$CONFIG" "${OVERRIDES[@]}"

echo "[5/5] Build similarity index..."
python -m src.index --config "$CONFIG" "${OVERRIDES[@]}"

echo ""
echo "Monitor complete. Results in data/output/"
echo "   Run: streamlit run app/dashboard.py"
