import base64
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Frame:
    t: float
    jpeg_b64: str


def sample(path: Path, fps: float = 2.0, long_side: int = 848) -> tuple[list[Frame], float]:
    """Кадры с шагом 1/fps (последовательное чтение, без seek). Возвращает кадры и длительность."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(src / fps))
    frames, i = [], 0
    while cap.grab():
        if i % step == 0:
            ok, fr = cap.retrieve()
            if ok:
                frames.append(Frame(t=i / src, jpeg_b64=encode(resize(fr, long_side))))
        i += 1
    cap.release()
    return frames, (n / src if n > 0 else (frames[-1].t if frames else 0.0))


def resize(frame: np.ndarray, long_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    s = min(1.0, long_side / max(h, w))
    nw, nh = max(32, round(w * s) // 32 * 32), max(32, round(h * s) // 32 * 32)
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def encode(frame: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def grab(path: Path, times: list[float], long_side: int | None = 960) -> list[tuple[float, np.ndarray]]:
    """Точные кадры для доказательств."""
    cap = cv2.VideoCapture(str(path))
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
        ok, fr = cap.read()
        if ok:
            out.append((t, resize(fr, long_side) if long_side else fr))
    cap.release()
    return out
