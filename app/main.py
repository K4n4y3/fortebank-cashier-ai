import asyncio
import json
import logging
import secrets
import time
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import config, rules
from app.jobs import JobStore, worker
from app.pipeline import get_pipeline
from app.pipeline.decode import probe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

store = JobStore(config.DATA_DIR)


async def _cleanup_loop() -> None:
    while True:
        removed = store.cleanup(config.RETENTION_HOURS * 3600)
        if removed:
            log.info("retention: removed %d old jobs", removed)
        await asyncio.sleep(1800)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not config.API_KEY:
        log.warning("API_KEY не задан — все запросы к /api/v1 будут отклонены")
    store.queue = asyncio.Queue()
    for job_id in store.load_existing():
        store.queue.put_nowait(job_id)
    # Прогрев моделей до первого запроса жюри
    warmup = getattr(get_pipeline(config.PIPELINE), "warmup", None)
    if warmup:
        log.info("warming up pipeline %s", config.PIPELINE)
        await asyncio.get_running_loop().run_in_executor(None, warmup)
    tasks = [asyncio.create_task(worker(store)) for _ in range(config.WORKERS)]
    if config.RETENTION_HOURS > 0:
        tasks.append(asyncio.create_task(_cleanup_loop()))
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(
    title="ForteBank Cash Discipline AI",
    version="1.0",
    lifespan=lifespan,
    description="API для проверки видеозаписей кассы (раздел 10 ТЗ). Ключ — в заголовке X-API-Key.",
)


def _key_ok(key: str) -> bool:
    return bool(config.API_KEY) and secrets.compare_digest(key.encode(), config.API_KEY.encode())


def require_key(request: Request) -> None:
    if not _key_ok(request.headers.get("X-API-Key", "")):
        raise HTTPException(status_code=401, detail="Неверный или отсутствующий X-API-Key")


def require_ui(request: Request) -> None:
    if config.UI_OPEN:
        return
    if not _key_ok(request.headers.get("X-API-Key") or request.query_params.get("key", "")):
        raise HTTPException(status_code=401, detail="Требуется ключ доступа")


async def _accept_upload(file: UploadFile, source: str) -> dict:
    name = file.filename or "video.mov"
    suffix = Path(name).suffix.lower()
    if suffix not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Неподдерживаемый формат {suffix or '(без расширения)'}; ожидается .mov")
    job = store.create(name, suffix, source)
    size = 0
    try:
        with open(job.input_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > config.MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=f"Файл больше {config.MAX_UPLOAD_BYTES // 2**20} МБ")
                out.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Пустой файл")
        # битый или не видео — отвечаем сразу, а не через failed спустя время
        try:
            await asyncio.get_running_loop().run_in_executor(None, probe, Path(job.input_path))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
    except BaseException:
        store.discard(job)
        raise
    store.save(job)
    assert store.queue is not None
    store.queue.put_nowait(job.job_id)
    log.info("job %s accepted (%s): %s (%.1f MB)", job.job_id, source, name, size / 1e6)
    return {"job_id": job.job_id, "status": job.status}


def _job_or_404(job_id: str, source: str | None = None):
    job = store.get(job_id)
    # интерфейс видит только свои задания: видео жюри (загруженные по ключу) через открытый UI не достать
    if not job or (source and job.source != source):
        raise HTTPException(status_code=404, detail="Задание не найдено")
    return job


_model_state = {"checked": 0.0, "ok": None}


def _model_ready() -> bool | None:
    """Отвечает ли vLLM (проверка не чаще раза в 10 с). None — конвейер без модели (dummy)."""
    if config.PIPELINE != "full":
        return None
    if time.time() - _model_state["checked"] > 10:
        from app.pipeline.full import VLLM_API, VLLM_MODEL
        try:
            with urllib.request.urlopen(f"{VLLM_API}/v1/models", timeout=3) as r:
                _model_state["ok"] = VLLM_MODEL in [m["id"] for m in json.loads(r.read())["data"]]
        except Exception:  # noqa: BLE001
            _model_state["ok"] = False
        _model_state["checked"] = time.time()
    return _model_state["ok"]


# ---------- Публичный API для жюри (раздел 10 ТЗ) ----------

@app.post("/api/v1/analyze", dependencies=[Depends(require_key)])
async def analyze(file: UploadFile = File(...)):
    return await _accept_upload(file, "api")


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
async def get_job(job_id: str):
    job = _job_or_404(job_id)
    return job.api_view(store.queue_position(job))


@app.get("/api/v1/health")
async def health():
    ready = await asyncio.get_running_loop().run_in_executor(None, _model_ready)
    return {
        "status": "ok" if ready is not False else "model_unavailable",
        "pipeline": config.PIPELINE,
        "model_ready": ready,
        "queue": store.queue.qsize() if store.queue else 0,
        "workers": config.WORKERS,
    }


# ---------- API веб-интерфейса ----------

@app.post("/ui-api/analyze", dependencies=[Depends(require_ui)])
async def ui_analyze(file: UploadFile = File(...)):
    return await _accept_upload(file, "ui")


@app.get("/ui-api/jobs/{job_id}", dependencies=[Depends(require_ui)])
async def ui_job(job_id: str):
    job = _job_or_404(job_id, "ui")
    return job.ui_view(store.queue_position(job))


@app.get("/ui-api/jobs/{job_id}/video", dependencies=[Depends(require_ui)])
async def ui_video(job_id: str):
    job = _job_or_404(job_id, "ui")
    # .mov с H.264 браузеры воспроизводят, если отдать его как mp4
    return FileResponse(job.input_path, media_type="video/mp4")


@app.get("/ui-api/types", dependencies=[Depends(require_ui)])
async def ui_types():
    return rules.catalog()


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": "Ожидается видеофайл в поле file (multipart/form-data)"})


app.mount("/static", StaticFiles(directory=config.WEB_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(config.WEB_DIR / "index.html")
