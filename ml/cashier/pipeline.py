"""Точка входа ML-части: analyze(video) -> результат в формате раздела 10 ТЗ.

Схема (двухступенчатая, одна модель Qwen3-VL-30B-A3B через локальный vLLM):
  1. detect  — VLM просматривает ролик окнами по 10 с и предлагает кандидатов (тип, время, наблюдение);
  2. verify  — кандидаты «рискованных» типов перепроверяются узким вопросом на увеличенных кадрах;
  3. evidence — 1–3 JPEG-кадра вокруг таймкода, описание на русском + тип + норматив из каталога.

Если до дедлайна остаётся мало времени, шаг 2 пропускается (лучше лишняя тревога, чем пропуск).
"""
import base64
import concurrent.futures as cf
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from ml.cashier import catalog
from ml.cashier.detect import detect
from ml.cashier.verify import verify
from ml.cashier.video import grab, resize


@dataclass
class Config:
    api: str = "http://127.0.0.1:8000"
    model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
    fps: float = 2.0
    window: float = 10.0
    stride: float = 5.0
    long_side: int = 848
    parallel: int = 16
    thr: float = 0.5
    # политика второй ступени: какие типы перепроверять и какие не выдавать вовсе
    verify_types: set = field(default_factory=lambda: {"T1", "T4", "T5", "T8", "T9", "T13"})
    drop_types: set = field(default_factory=set)
    deadline_sec: float = 240.0      # бюджет на ролик; жюри ждёт 300 с
    # если на проверку осталось меньше — отдаём кандидатов без неё; не успевшие проверку кандидаты тоже остаются
    verify_reserve_sec: float = 20.0


def _duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Не удалось открыть видеофайл: формат не поддерживается или файл повреждён")
    d = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    return d


def _evidence(path: Path, t: float, dur: float) -> list[dict]:
    times = []
    for off in (0.0, -1.5, 1.5):  # кадр события первым, затем контекст до/после
        tt = round(min(max(0.0, t + off), max(0.0, dur - 0.05)), 2)
        if all(abs(tt - x) > 0.3 for x in times):
            times.append(tt)
    out = []
    for tt, fr in grab(path, times, long_side=None):
        ok, buf = cv2.imencode(".jpg", resize(fr, 960), [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            out.append({"timestamp_sec": tt, "image_base64": base64.b64encode(buf.tobytes()).decode("ascii")})
    return out[:3]


# Слова, по которым наблюдение модели выходит за рамки видимого (р. 7 ТЗ): оформление, суммы, намерения,
# обвинения. Такое наблюдение в описание не попадает — остаются нейтральное пояснение и норматив из каталога.
_UNVERIFIABLE = ("оформ", "сумм", "намерен", "умышл", "специально", "чтобы скрыть", "хищ", "укра", "кража",
                 "мошен", "незакон", "присво")


def observation_ok(observation: str) -> bool:
    low = observation.lower()
    return not any(w in low for w in _UNVERIFIABLE)


def _description(type_id: str, observation: str) -> str:
    r = catalog.rule(type_id)
    obs = observation.strip().rstrip(".") if observation_ok(observation) else ""
    parts = [f"На видео: {obs}." if obs else "", r["why"], catalog.source_text(type_id)]
    return " ".join(p for p in parts if p)


def analyze(video: str | Path, cfg: Config | None = None, progress=None) -> dict:
    """Возвращает {"verdict", "violations": [поля ТЗ], "details": [доп. поля для UI], "timing": {...}}.

    В API жюри отдаются только verdict и violations; details — для интерфейса (тип-код, уверенность,
    источники, как подтверждено).
    """
    cfg = cfg or Config()
    path = Path(video)
    t0 = time.time()
    say = progress or (lambda frac, stage: None)
    left = lambda: cfg.deadline_sec - (time.time() - t0)  # noqa: E731
    dur = _duration(path)

    say(0.05, "Поиск возможных нарушений")
    # ответ модели на окно ждём не дольше бюджета ролика (10 с — на кадры-доказательства)
    cands, _ = detect(path, cfg.api, cfg.model, cfg.fps, cfg.window, cfg.stride, cfg.long_side, cfg.parallel, cfg.thr,
                      timeout=max(20.0, left() - 10.0))
    cands = [c for c in cands if c["type"] not in cfg.drop_types]
    t_detect = time.time() - t0

    say(0.6, "Проверка найденных эпизодов")
    kept, skipped_verify = [], False
    to_verify = [c for c in cands if c["type"] in cfg.verify_types]
    kept += [dict(c, verified="not_required") for c in cands if c["type"] not in cfg.verify_types]
    if to_verify and left() < cfg.verify_reserve_sec:
        skipped_verify = True
        kept += [dict(c, verified="skipped_deadline") for c in to_verify]
    elif to_verify:
        ex = cf.ThreadPoolExecutor(8)
        futs = [(c, ex.submit(verify, path, c["type"], c["t"], dur, cfg.api, cfg.model)) for c in to_verify]
        cf.wait([f for _, f in futs], timeout=max(5.0, left() - 10.0))
        for c, f in futs:
            if not f.done():
                # проверка не успела к дедлайну — кандидата не теряем (пропуск хуже лишней тревоги)
                skipped_verify = True
                kept.append(dict(c, verified="skipped_deadline"))
            elif f.result().confirmed:
                kept.append(dict(c, verified="confirmed", verify_evidence=f.result().evidence))
        ex.shutdown(wait=False, cancel_futures=True)
    t_verify = time.time() - t0 - t_detect

    say(0.85, "Подготовка кадров-доказательств")
    kept.sort(key=lambda c: c["t"])
    violations, details = [], []
    for c in kept:
        r = catalog.rule(c["type"])
        violations.append({
            "type": r["name"],
            "timecode_sec": round(float(c["t"]), 1),
            "description": _description(c["type"], c.get("obs", "")),
            "evidence_frames": _evidence(path, float(c["t"]), dur),
        })
        obs = c.get("obs", "")
        details.append({
            "type_id": c["type"], "confidence": round(float(c["conf"]), 2),
            "observation": obs.strip() if observation_ok(obs) else "",  # то же, что попало в description
            "why": r["why"], "sources": r.get("sources") or [], "verified": c["verified"],
        })
    say(1.0, "Готово")
    return {
        "verdict": "violation" if violations else "no_violation",
        "violations": violations,
        "details": details,
        "timing": {"duration_sec": round(dur, 1), "detect_sec": round(t_detect, 1), "verify_sec": round(t_verify, 1),
                   "total_sec": round(time.time() - t0, 1), "verify_skipped": skipped_verify},
    }


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Анализ одного видео: python -m ml.cashier.pipeline video.mov")
    ap.add_argument("video")
    ap.add_argument("--api", default=Config.api)
    ap.add_argument("--model", default=Config.model)
    a = ap.parse_args()
    res = analyze(a.video, Config(api=a.api, model=a.model))
    for v in res["violations"]:
        for f in v["evidence_frames"]:
            f["image_base64"] = f["image_base64"][:32] + "..."
    print(json.dumps(res, ensure_ascii=False, indent=1))
