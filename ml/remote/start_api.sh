#!/usr/bin/env bash
# (Пере)запуск API-сервера на поде: останавливает старый процесс и стартует новый на 0.0.0.0:8080.
#   bash ml/remote/start_api.sh            # из корня репозитория на поде (там же лежит .env)
# Лог: /workspace/logs/api.log; проверка: curl -s 127.0.0.1:8080/api/v1/health
set -euo pipefail
cd "$(dirname "$0")/../.."
source /workspace/env.sh
pkill -f "uvicorn app.main:app" || true
sleep 2
nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 >> /workspace/logs/api.log 2>&1 < /dev/null &
for i in $(seq 1 30); do
  sleep 1
  if curl -sf 127.0.0.1:8080/api/v1/health; then echo; exit 0; fi
done
echo "API не поднялся, см. /workspace/logs/api.log" >&2
exit 1
