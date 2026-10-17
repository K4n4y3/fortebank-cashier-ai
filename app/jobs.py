import asyncio
import json
import logging
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app import config
from app.pipeline import get_pipeline

log = logging.getLogger("jobs")


@dataclass
class Job:
    job_id: str
    filename: str
    input_path: str
    created_at: float = field(default_factory=time.time)
    status: str = "processing"  # processing | done | failed
    stage: str = "В очереди"
    progress: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    verdict: str | None = None
    duration_sec: float | None = None
    processing_sec: float | None = None
    notes: list[str] = field(default_factory=list)
    violations_api: list[dict] = field(default_factory=list)
    violations_ui: list[dict] = field(default_factory=list)

    def api_view(self) -> dict:
        if self.status == "done":
            return {
                "job_id": self.job_id,
                "status": "done",
                "verdict": self.verdict,
                "violations": self.violations_api,
            }
        d = {"job_id": self.job_id, "status": self.status, "progress": round(self.progress, 2), "stage": self.stage}
        if self.error:
            d["error"] = self.error
        return d

    def ui_view(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 2),
            "filename": self.filename,
            "error": self.error,
            "verdict": self.verdict,
            "duration_sec": self.duration_sec,
            "processing_sec": self.processing_sec,
            "violations": self.violations_ui,
        }


class JobStore:
    def __init__(self, root: Path):
        self.root = root / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.queue: asyncio.Queue[str] | None = None

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def save(self, job: Job) -> None:
        path = self.job_dir(job.job_id) / "job.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(job), ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def load_existing(self) -> list[str]:
        """Восстановление после рестарта: незавершённые задания снова ставятся в очередь."""
        pending = []
        for f in self.root.glob("*/job.json"):
            try:
                job = Job(**json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                log.exception("cannot load %s", f)
                continue
            self.jobs[job.job_id] = job
            if job.status == "processing":
                pending.append(job.job_id)
        return pending

    def create(self, filename: str, suffix: str) -> Job:
        job_id = uuid.uuid4().hex[:12]
        d = self.job_dir(job_id)
        d.mkdir(parents=True)
        job = Job(job_id=job_id, filename=filename, input_path=str(d / f"input{suffix}"))
        with self.lock:
            self.jobs[job_id] = job
        return job

    def discard(self, job: Job) -> None:
        with self.lock:
            self.jobs.pop(job.job_id, None)
        shutil.rmtree(self.job_dir(job.job_id), ignore_errors=True)

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def update(self, job: Job, **fields) -> None:
        with self.lock:
            for k, v in fields.items():
                setattr(job, k, v)
        self.save(job)


def _run_job(store: JobStore, job: Job) -> None:
    analyze = get_pipeline(config.PIPELINE).analyze
    deadline = job.created_at + config.JOB_DEADLINE_SEC
    store.update(job, started_at=time.time(), stage="Анализ видео", progress=0.01)

    last_save = [0.0]

    def progress(frac: float, stage: str) -> None:
        job.progress = max(job.progress, min(0.99, frac))
        job.stage = stage
        now = time.monotonic()
        if now - last_save[0] > 1.0:
            last_save[0] = now
            store.save(job)

    try:
        result = analyze(Path(job.input_path), progress=progress, deadline=deadline)
    except Exception as e:  # noqa: BLE001 — любая ошибка конвейера = failed с понятным текстом
        log.error("job %s failed: %s", job.job_id, traceback.format_exc())
        store.update(job, status="failed", error=str(e) or e.__class__.__name__, stage="Ошибка", finished_at=time.time())
        return
    store.update(
        job,
        status="done",
        stage="Готово",
        progress=1.0,
        verdict=result.verdict,
        duration_sec=round(result.duration_sec, 2),
        processing_sec=round(result.processing_sec, 2),
        notes=result.notes,
        violations_api=[v.to_api() for v in result.violations],
        violations_ui=[v.to_ui() for v in result.violations],
        finished_at=time.time(),
    )
    log.info("job %s done: %s, %d violations, %.1fs", job.job_id, result.verdict, len(result.violations), result.processing_sec)


async def worker(store: JobStore) -> None:
    assert store.queue is not None
    loop = asyncio.get_running_loop()
    while True:
        job_id = await store.queue.get()
        job = store.get(job_id)
        try:
            if job and job.status == "processing":
                await loop.run_in_executor(None, _run_job, store, job)
        finally:
            store.queue.task_done()
