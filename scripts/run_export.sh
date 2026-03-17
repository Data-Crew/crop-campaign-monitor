#!/bin/bash
set -euo pipefail

CONFIG=${1:-config/train.yaml}

if [ -n "${GPU_NAME:-}" ]; then
  export CUDA_VISIBLE_DEVICES=$(python -c "from src.gpu import resolve_gpu; print(resolve_gpu('${GPU_NAME}'))")
elif [ -n "${GPU_ID:-}" ]; then
  export CUDA_VISIBLE_DEVICES=${GPU_ID}
else
  export CUDA_VISIBLE_DEVICES=0
fi

echo "Exporting encoder from fine-tuned checkpoint..."
python -m train.export_encoder --config $CONFIG
echo "Export complete."
