#!/usr/bin/env bash
# Запуск vLLM на поде (только localhost — наружу не открываем).
#   bash /workspace/forte/ml/remote/serve_vllm.sh [model]
# Лог: /workspace/logs/vllm.log; готовность: curl -s 127.0.0.1:8000/v1/models
set -euo pipefail
source /workspace/env.sh
MODEL=${1:-Qwen/Qwen3-VL-8B-Instruct}
MAXLEN=${MAXLEN:-32768}
GPU_UTIL=${GPU_UTIL:-0.85}
pkill -f "vllm serve" || true
sleep 5
nohup vllm serve "$MODEL" \
  --host 127.0.0.1 --port 8000 \
  --served-model-name "$MODEL" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --limit-mm-per-prompt '{"image": 32, "video": 0}' \
  --max-num-seqs 32 \
  ${EXTRA:-} \
  > /workspace/logs/vllm.log 2>&1 &
echo "vllm starting: $MODEL (pid $!)"
