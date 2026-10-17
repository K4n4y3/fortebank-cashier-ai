import base64
from pathlib import Path

import cv2
import numpy as np

from app.pipeline.decode import grab_frames
from app.pipeline.types import EvidenceFrame

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 в долях кадра


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def draw_boxes(frame: np.ndarray, boxes: list[tuple[Box, str]]) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    for (x1, y1, x2, y2), label in boxes:
        p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        cv2.rectangle(out, p1, p2, (0, 0, 255), 2)
        if label:
            cv2.putText(out, label, (p1[0], max(14, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return out


def make_evidence(
    video: Path,
    timecode: float,
    duration: float,
    offsets: tuple[float, ...] = (0.0, -1.5, 1.5),
    boxes: dict[float, list[tuple[Box, str]]] | None = None,
) -> list[EvidenceFrame]:
    """Кадр события + контекст до/после (первым всегда кадр таймкода)."""
    times = []
    for off in offsets:
        t = min(max(0.0, timecode + off), max(0.0, duration - 0.05))
        if all(abs(t - x) > 0.3 for x in times):
            times.append(t)
    frames = []
    for t, frame in grab_frames(video, times, max_side=960):
        b = (boxes or {}).get(t)
        if b:
            frame = draw_boxes(frame, b)
        frames.append(EvidenceFrame(timestamp_sec=t, image_base64=encode_jpeg(frame)))
    return frames
