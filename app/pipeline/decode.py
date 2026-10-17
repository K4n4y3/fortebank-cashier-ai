from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def _open(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Не удалось открыть видеофайл: формат не поддерживается или файл повреждён")
    return cap


def probe(path: Path) -> VideoInfo:
    cap = _open(path)
    try:
        info = VideoInfo(
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()
    if info.frame_count <= 0:
        raise ValueError("Видео не содержит кадров")
    return info


def _resize(frame: np.ndarray, max_side: int | None) -> np.ndarray:
    if not max_side:
        return frame
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return frame
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def sample_frames(path: Path, fps: float, max_side: int | None = None) -> list[tuple[float, np.ndarray]]:
    """Последовательное чтение с прореживанием до `fps` кадров/с (быстрее, чем seek)."""
    cap = _open(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / fps))
    out: list[tuple[float, np.ndarray]] = []
    idx = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    out.append((idx / src_fps, _resize(frame, max_side)))
            idx += 1
    finally:
        cap.release()
    return out


def grab_frames(path: Path, times: list[float], max_side: int | None = None) -> list[tuple[float, np.ndarray]]:
    """Точные кадры полного разрешения для кадров-доказательств."""
    cap = _open(path)
    out = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
            ok, frame = cap.read()
            if ok:
                out.append((t, _resize(frame, max_side)))
    finally:
        cap.release()
    return out
