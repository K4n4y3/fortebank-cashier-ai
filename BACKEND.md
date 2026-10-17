# Передача бэкенду: что готово и как запустить

Репозиторий: https://github.com/K4n4y3/fortebank-cashier-ai (приватный).

Этот документ — для того, кто делает серверную часть и интерфейс. ML-часть подробно описана в
[ml/README.md](ml/README.md).

## 1. Что уже есть в репозитории

| Часть | Где | Состояние |
|---|---|---|
| **ML-модуль** — `analyze(видео) → результат по схеме ТЗ` | `ml/cashier/pipeline.py` | готов, проверен на 39 роликах и в нагрузке (5 роликов за 115 с) |
| **Каталог нарушений + нормативы** | `ml/cashier/violations.yaml` | 13 типов, пункты Правил НБРК № 231 и № 14 |
| **Черновик сервера** — FastAPI: API по разделу 10 ТЗ, очередь заданий, ключ `X-API-Key` | `app/` | работает; проверен локально в режиме `dummy` (контракт API, 401 без ключа, формат JSON, кадры base64) |
| **Черновик интерфейса** — загрузка, прогресс, вердикт, плеер с переходом к таймкоду, карточки с кадрами и нормативами | `web/` | работает, минимальный — можно доработать или заменить |
| Скрипты GPU-сервера | `ml/remote/setup_pod.sh`, `ml/remote/serve_vllm.sh` | готовы |

Черновики `app/` и `web/` — стартовая точка: их можно полностью переписать, главное — сохранить контракт
API из раздела 10 ТЗ.

**Проверено на поде через публичный адрес** (`PIPELINE=full`, 18.10.2026): `health` → ok, без ключа → 401,
`X-17.mov` → `done` за 8 с с нарушениями и кадрами; **5 самых длинных роликов, отправленных разом, готовы за
62–121 с** (лимит 300 с, `WORKERS=3`). Развёрнутая копия лежит на поде в `/workspace/forte-app`
(ключ — в `/workspace/forte-app/.env`, в git его нет).

## 2. Схема на GPU-сервере (RunPod, NVIDIA L40S 48 ГБ)

```
интернет ──► https://<POD_ID>-8080.proxy.runpod.net ──► FastAPI (0.0.0.0:8080)  [app/]
                                                           │ вызывает ml.cashier.pipeline.analyze()
                                                           ▼
                                                        vLLM (127.0.0.1:8000, наружу закрыт)
                                                           └─ Qwen3-VL-30B-A3B-Instruct-FP8, офлайн
```

- Публичный адрес RunPod обрывает запрос, если сервер не ответил за **100 с** — поэтому `POST /api/v1/analyze`
  сразу возвращает `job_id`, а результат забирается через `GET /api/v1/jobs/{id}` (так и сделано).
- Сервер должен слушать **0.0.0.0**, не 127.0.0.1, иначе прокси RunPod его не увидит.

## 3. Локальная разработка без GPU (режим `dummy`)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Linux/Mac: .venv/bin/python
cp .env.example .env                                          # и поменять API_KEY
.venv/Scripts/python -m uvicorn app.main:app --port 8080
```

Открыть http://127.0.0.1:8080. В режиме `PIPELINE=dummy` на любое видео возвращается одно тестовое
нарушение с настоящими кадрами из ролика — для отладки API и интерфейса.

## 4. С настоящей моделью со своего ноутбука (под включён)

1. Добавить свой SSH-ключ в аккаунт RunPod (**Settings → SSH Public Keys**) или получить доступ от владельца.
2. Пробросить порт vLLM к себе:
   ```bash
   ssh -N -L 8000:127.0.0.1:8000 root@<IP> -p <SSH_PORT> -i ~/.ssh/id_ed25519
   ```
   (`<IP>` и `<SSH_PORT>` — во вкладке **Connect → SSH over exposed TCP** пода; после перезапуска пода порт может смениться.)
3. В `.env`: `PIPELINE=full`, `VLLM_API=http://127.0.0.1:8000` — и запустить сервер как в п. 3.
   Видео читается на ноутбуке, на под уходят только кадры по SSH-туннелю.

## 5. Деплой на под (для жюри)

```bash
ssh root@<IP> -p <SSH_PORT> -i ~/.ssh/id_ed25519
cd /workspace && git clone https://github.com/K4n4y3/fortebank-cashier-ai.git forte-app   # или git pull, если уже склонирован
cd forte-app
bash ml/remote/setup_pod.sh                 # один раз: venv в /workspace, vLLM, веса модели (~3–5 мин)
bash ml/remote/serve_vllm.sh                # запуск модели офлайн (~2–3 мин до готовности)
until curl -sf 127.0.0.1:8000/v1/models; do sleep 5; done

cp .env.example .env && nano .env           # PIPELINE=full, API_KEY=<ключ для жюри>, DATA_DIR=/workspace/jobs
source /workspace/env.sh
nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 > /workspace/logs/api.log 2>&1 < /dev/null &
```

Проверка снаружи (с любого компьютера):

```bash
curl -s https://<POD_ID>-8080.proxy.runpod.net/api/v1/health
curl -s -X POST https://<POD_ID>-8080.proxy.runpod.net/api/v1/analyze -H "X-API-Key: <ключ>" -F "file=@X-17.mov"
curl -s https://<POD_ID>-8080.proxy.runpod.net/api/v1/jobs/<job_id> -H "X-API-Key: <ключ>"
```

`<POD_ID>` — в карточке пода; на поде также `echo $RUNPOD_POD_ID`.
Порт **8080** должен быть в **Expose HTTP Ports** пода (он там уже есть).

## 6. Контракт

### API жюри (раздел 10 ТЗ) — `app/main.py`

| Метод | Путь | Что делает |
|---|---|---|
| POST | `/api/v1/analyze` | поле `file` (.mov), заголовок `X-API-Key` → `{"job_id", "status": "processing"}` |
| GET | `/api/v1/jobs/{job_id}` | `processing` / `failed` (+ `error`) / `done` → `{"job_id","status","verdict","violations":[{"type","timecode_sec","description","evidence_frames":[{"timestamp_sec","image_base64"}]}]}` |
| GET | `/api/v1/health` | без ключа: статус и длина очереди |

Без верного ключа — `401`. Неподдерживаемое расширение — `400`, пустой файл — `400`, > `MAX_UPLOAD_MB` — `413`.

### Вызов модели напрямую (если пишете свой сервер)

```python
from ml.cashier.pipeline import Config, analyze

r = analyze("/path/input.mov",
            Config(api="http://127.0.0.1:8000", deadline_sec=240),
            progress=lambda frac, stage: ...)      # 0..1 и подпись этапа — для полосы прогресса
r["verdict"], r["violations"]     # ← отдавать жюри как есть
r["details"]                      # по тому же индексу: type_id, confidence, sources[], verified — для интерфейса
r["timing"]                       # время этапов — для логов
```

- Функция блокирующая (5–30 с на ролик); вызывать в отдельном потоке/процессе. Несколько вызовов
  одновременно — нормально: vLLM сам объединяет запросы. В черновике — `WORKERS=3` потока.
- Если модель недоступна — `RuntimeError("VLM недоступен ...")`; если видео не открывается — `ValueError`.
  В обоих случаях задаче ставится `failed` с текстом ошибки, а не ложный «нарушений нет».
- Зависимости ML-модуля для сервера — только `opencv-python-headless`, `numpy`, `pyyaml` (torch не нужен:
  модель работает внутри vLLM).

## 7. Переменные окружения (`.env`, в git не попадает)

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `API_KEY` | — | ключ жюри; сгенерировать: `python -c "import secrets; print(secrets.token_urlsafe(24))"` |
| `PIPELINE` | `full` | `full` — модель, `dummy` — заглушка |
| `VLLM_API` / `VLLM_MODEL` | `http://127.0.0.1:8000` / `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` | где модель |
| `WORKERS` | `1` (в примере `3`) | сколько роликов параллельно |
| `JOB_DEADLINE_SEC` | `270` | бюджет на ролик от загрузки; при нехватке времени модель пропускает проверку, но отвечает |
| `DATA_DIR` | `./data` | куда сохраняются загруженные видео и результаты |
| `UI_OPEN` | `1` | интерфейс без ключа (FAQ 27). Учтите: любой, у кого есть адрес, может загружать видео |

## 8. Правила, которые нельзя нарушать

- **Тестовые видео жюри** (лежат в `DATA_DIR`) нельзя использовать для обучения, публиковать, передавать
  третьим лицам и показывать любым внешним ИИ-сервисам (р. 10 ТЗ).
- Никаких облачных AI-API в обработке (р. 7) — всё уже работает внутри сервиса.
- Ключ жюри и адрес/порт SSH пода — не в git.
- После хакатона удалить данные: под (Terminate), `DATA_DIR`, `dataset/`.

## 9. Чек-лист перед сдачей

- [ ] Под включён на всё окно проверки жюри; баланс RunPod с запасом (~$26 в сутки).
- [ ] `serve_vllm.sh` запущен, `curl 127.0.0.1:8000/v1/models` отвечает.
- [ ] Сервер на `0.0.0.0:8080`, `PIPELINE=full`, свой `API_KEY`.
- [ ] Снаружи: `health` → ok; без ключа → 401; ролик из датасета → `done` с нарушениями за < 60 с.
- [ ] Жюри передан адрес `https://<POD_ID>-8080.proxy.runpod.net`, адрес API и ключ.
