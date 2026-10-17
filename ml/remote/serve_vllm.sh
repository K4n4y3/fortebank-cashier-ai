#!/usr/bin/env bash
# Запуск vLLM на поде (только localhost — наружу не открываем).
#   bash /workspace/forte/ml/remote/serve_vllm.sh [model]
# Лог: /workspace/logs/vllm.log; готовность: curl -s 127.0.0.1:8000/v1/models
#
# Сервис работает без доступа в интернет (требование ТЗ, р. 7): веса берутся только из локального
# кэша (их заранее скачивает setup_pod.sh), обращения к Hugging Face и сбор статистики vLLM отключены.
set -euo pipefail
source /workspace/env.sh
MODEL=${1:-Qwen/Qwen3-VL-30B-A3B-Instruct-FP8}
MAXLEN=${MAXLEN:-16384}
GPU_UTIL=${GPU_UTIL:-0.92}

export HF_HUB_OFFLINE=1          # только локальный кэш весов, без запросов к huggingface.co
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1     # без анонимной статистики использования vLLM
export DO_NOT_TRACK=1

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
