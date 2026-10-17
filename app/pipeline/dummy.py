"""Отладочный конвейер для проверки API и UI без GPU. В продакшене не используется."""
import time
from pathlib import Path

from app.pipeline.decode import probe
from app.pipeline.evidence import make_evidence
from app.pipeline.types import AnalysisResult, ProgressFn, Violation


def analyze(video: Path, progress: ProgressFn, deadline: float) -> AnalysisResult:
    started = time.monotonic()
    info = probe(video)
    for i in range(5):
        progress(i / 5, "Отладочный режим: имитация анализа")
        time.sleep(0.3)
    t = round(info.duration_sec * 0.4, 1)
    v = Violation(
        type_id="T1",
        timecode_sec=t,
        observation="[ОТЛАДКА] Тестовое событие для проверки интерфейса",
        confidence=0.5,
        evidence=make_evidence(video, t, info.duration_sec),
        sources_hint=["dummy"],
    )
    return AnalysisResult(
        violations=[v],
        duration_sec=info.duration_sec,
        processing_sec=time.monotonic() - started,
        notes=["dummy pipeline"],
    )
