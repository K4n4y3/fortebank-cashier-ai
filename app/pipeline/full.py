"""Боевой конвейер: вызывает ML-модуль ml.cashier.pipeline.analyze (нужен запущенный vLLM).

Переменные окружения:
  VLLM_API   — адрес vLLM (по умолчанию http://127.0.0.1:8000; с ноутбука — через ssh -L 8000:127.0.0.1:8000)
  VLLM_MODEL — имя модели в vLLM (по умолчанию Qwen/Qwen3-VL-30B-A3B-Instruct-FP8)
"""
import json
import logging
import os
import time
import urllib.request
from pathlib import Path

from app.pipeline.types import AnalysisResult, ProgressFn
from ml.cashier.pipeline import Config
from ml.cashier.pipeline import analyze as ml_analyze

log = logging.getLogger("pipeline.full")

VLLM_API = os.environ.get("VLLM_API", "http://127.0.0.1:8000")
VLLM_MODEL = os.environ.get("VLLM_MODEL", Config.model)


class _Item:
    """Нарушение в том виде, как его ждёт jobs.py: to_api() — поля ТЗ, to_ui() — плюс детали для интерфейса."""

    def __init__(self, api: dict, details: dict):
        self.api, self.details = api, details

    def to_api(self) -> dict:
        return self.api

    def to_ui(self) -> dict:
        return {**self.api, **self.details}


def analyze(video: Path, progress: ProgressFn, deadline: float) -> AnalysisResult:
    # deadline — абсолютное время (epoch), к которому задание должно быть готово; оставляем 10 с запаса
    budget = max(30.0, deadline - time.time() - 10.0)
    cfg = Config(api=VLLM_API, model=VLLM_MODEL, deadline_sec=budget)
    r = ml_analyze(video, cfg, progress=progress)
    t = r["timing"]
    notes = [f"detect {t['detect_sec']}s, verify {t['verify_sec']}s"] + (["verify skipped (deadline)"] if t["verify_skipped"] else [])
    return AnalysisResult(
        violations=[_Item(v, d) for v, d in zip(r["violations"], r["details"])],
        duration_sec=t["duration_sec"],
        processing_sec=t["total_sec"],
        notes=notes,
    )


def warmup() -> None:
    """Проверка при старте сервера: доступен ли vLLM и загружена ли нужная модель."""
    try:
        with urllib.request.urlopen(f"{VLLM_API}/v1/models", timeout=5) as r:
            models = [m["id"] for m in json.loads(r.read())["data"]]
        if VLLM_MODEL not in models:
            log.warning("vLLM доступен, но модели %s нет (есть: %s)", VLLM_MODEL, models)
        else:
            log.info("vLLM готов: %s", VLLM_MODEL)
    except Exception as e:  # noqa: BLE001 — сервер всё равно стартует, задания получат failed с понятной ошибкой
        log.warning("vLLM недоступен по адресу %s: %s", VLLM_API, e)
