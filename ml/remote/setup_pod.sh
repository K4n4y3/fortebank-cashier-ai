#!/usr/bin/env bash
# Подготовка GPU-пода RunPod: окружение в /workspace (переживает Stop), vLLM, веса VLM.
# Запуск на поде:  bash /workspace/forte/ml/remote/setup_pod.sh  (лог: /workspace/logs/setup.log)
set -euo pipefail

MODELS=${MODELS:-"Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"}

mkdir -p /workspace/logs /workspace/hf
cat > /workspace/env.sh <<'EOF'
export HF_HOME=/workspace/hf
export VLLM_CACHE_ROOT=/workspace/vllm_cache
export PATH=/workspace/venv/bin:$PATH
EOF
grep -q 'source /workspace/env.sh' ~/.bashrc || echo 'source /workspace/env.sh' >> ~/.bashrc
source /workspace/env.sh

if ! command -v uv >/dev/null; then
  pip install -q --break-system-packages uv
fi
[ -x /workspace/venv/bin/python ] || uv venv /workspace/venv --python 3.12
uv pip install --python /workspace/venv/bin/python vllm opencv-python-headless numpy "huggingface_hub[cli]"

for m in $MODELS; do
  /workspace/venv/bin/hf download "$m"
done

/workspace/venv/bin/python - <<'EOF'
import torch, vllm
print("VLLM_OK", vllm.__version__, torch.__version__, torch.cuda.is_available())
EOF
