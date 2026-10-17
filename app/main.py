import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import config, rules
from app.jobs import JobStore, worker
from app.pipeline import get_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

store = JobStore(config.DATA_DIR)


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
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(title="ForteBank Cash Discipline AI", version="1.0", lifespan=lifespan)


def require_key(request: Request) -> None:
    key = request.headers.get("X-API-Key", "")
    if not config.API_KEY or not secrets.compare_digest(key.encode(), config.API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Неверный или отсутствующий X-API-Key")


def require_ui(request: Request) -> None:
    if config.UI_OPEN:
        return
    key = request.headers.get("X-API-Key") or request.query_params.get("key", "")
    if not config.API_KEY or not secrets.compare_digest(key.encode(), config.API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Требуется ключ доступа")


async def _accept_upload(file: UploadFile) -> dict:
    name = file.filename or "video.mov"
    suffix = Path(name).suffix.lower()
    if suffix not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Неподдерживаемый формат {suffix or '(без расширения)'}; ожидается .mov")
    job = store.create(name, suffix)
    size = 0
    try:
        with open(job.input_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > config.MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Файл слишком большой")
                out.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Пустой файл")
    except BaseException:
        store.discard(job)
        raise
    store.save(job)
    assert store.queue is not None
    store.queue.put_nowait(job.job_id)
    log.info("job %s accepted: %s (%.1f MB)", job.job_id, name, size / 1e6)
    return {"job_id": job.job_id, "status": job.status}


def _job_or_404(job_id: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Задание не найдено")
    return job


# ---------- Публичный API для жюри (раздел 10 ТЗ) ----------

@app.post("/api/v1/analyze", dependencies=[Depends(require_key)])
async def analyze(file: UploadFile = File(...)):
    return await _accept_upload(file)


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
async def get_job(job_id: str):
    return _job_or_404(job_id).api_view()


@app.get("/api/v1/health")
async def health():
    q = store.queue.qsize() if store.queue else 0
    return {"status": "ok", "pipeline": config.PIPELINE, "queue": q}


# ---------- API веб-интерфейса ----------

@app.post("/ui-api/analyze", dependencies=[Depends(require_ui)])
async def ui_analyze(file: UploadFile = File(...)):
    return await _accept_upload(file)


@app.get("/ui-api/jobs/{job_id}", dependencies=[Depends(require_ui)])
async def ui_job(job_id: str):
    return _job_or_404(job_id).ui_view()


@app.get("/ui-api/jobs/{job_id}/video", dependencies=[Depends(require_ui)])
async def ui_video(job_id: str):
    job = _job_or_404(job_id)
    return FileResponse(job.input_path, media_type="video/quicktime" if job.input_path.endswith(".mov") else None)


@app.get("/ui-api/types", dependencies=[Depends(require_ui)])
async def ui_types():
    return rules.catalog()


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


app.mount("/static", StaticFiles(directory=config.WEB_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(config.WEB_DIR / "index.html")
