"""Память размеченных примеров (эксперимент v2, в продукт не вошёл — см. ml/README.md «Что пробовали»).

Эмбеддинги клипов V-JEPA 2 и поиск похожих эпизодов.

База строится по обучающим роликам: клипы вокруг размеченных нарушений получают тип, клипы нормальных
роликов — метку normal. Для кандидата первой ступени ищем ближайшие примеры и сравниваем, на что он больше
похож: на нарушения этого типа или на норму. Модель — V-JEPA 2 ViT-L (facebook/vjepa2-vitl-fpc64-256, MIT).

Клип — 4 с, 16 кадров. Кадр подаётся тремя квадратными видами: весь кадр с полями и две половины по длинной
стороне (у вертикального ролика — верх и низ), чтобы детали стола и двери не терялись при сжатии до 256 px.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ml.cashier.video import grab

MODEL = "facebook/vjepa2-vitl-fpc64-256"
CLIP_SEC = 4.0
FRAMES = 16
SIZE = 256
VIEWS = ("full", "half_a", "half_b")


def views(frame_bgr: np.ndarray) -> np.ndarray:
    """Кадр -> [3, SIZE, SIZE, 3] RGB: весь кадр с полями и две квадратные половины по длинной стороне."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    s = SIZE / max(h, w)
    small = cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    full = np.full((SIZE, SIZE, 3), 120, np.uint8)
    y0, x0 = (SIZE - small.shape[0]) // 2, (SIZE - small.shape[1]) // 2
    full[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    side = min(h, w)
    halves = [rgb[:side], rgb[h - side:]] if h >= w else [rgb[:, :side], rgb[:, w - side:]]
    halves = [cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_AREA) for c in halves]
    return np.stack([full, *halves])


def clip_times(center: float) -> list[float]:
    step = CLIP_SEC / FRAMES
    return [center - CLIP_SEC / 2 + (k + 0.5) * step for k in range(FRAMES)]


def clips_at(path: Path, centers: list[float], duration: float) -> np.ndarray:
    """Клипы вокруг заданных моментов (точный seek): [N, VIEWS, FRAMES, SIZE, SIZE, 3]."""
    out = []
    for c in centers:
        times = [min(max(0.0, t), max(0.0, duration - 0.05)) for t in clip_times(c)]
        frames = [fr for _, fr in grab(path, times, long_side=None)]
        if not frames:
            raise ValueError(f"no frames at {c:.1f}s in {path}")
        frames += [frames[-1]] * (FRAMES - len(frames))
        out.append(np.stack([views(f) for f in frames], axis=1))  # [VIEWS, FRAMES, S, S, 3]
    return np.stack(out)


def sampled_views(path: Path) -> np.ndarray:
    """Весь ролик с частотой клипа (FRAMES / CLIP_SEC кадров/с), последовательное чтение: [n, VIEWS, S, S, 3]."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = FRAMES / CLIP_SEC
    frames, i, next_t = [], 0, 0.0
    while cap.grab():
        if i / src + 1e-6 >= next_t:
            ok, fr = cap.retrieve()
            if ok:
                frames.append(views(fr))
            next_t += 1.0 / fps
        i += 1
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, len(VIEWS), SIZE, SIZE, 3), np.uint8)


def clip_centers(n_frames: int, stride: float = 1.0) -> np.ndarray:
    dur = n_frames * CLIP_SEC / FRAMES
    return np.arange(CLIP_SEC / 2, max(CLIP_SEC / 2, dur - CLIP_SEC / 2) + 1e-6, stride)


def clip_from(frames: np.ndarray, center: float) -> np.ndarray:
    """Клип из кадров sampled_views: [VIEWS, FRAMES, S, S, 3]."""
    k0 = int(round((center - CLIP_SEC / 2) * FRAMES / CLIP_SEC))
    idx = [min(max(0, k0 + k), len(frames) - 1) for k in range(FRAMES)]
    return frames[idx].transpose(1, 0, 2, 3, 4)


class Encoder:
    """V-JEPA 2: клипы -> L2-нормированные эмбеддинги (среднее по токенам) для каждого вида."""

    def __init__(self, model: str = MODEL, device: str | None = None):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(model, dtype=self.dtype).to(self.device).eval()
        # нормализация ImageNet, как в препроцессоре V-JEPA 2 (сам препроцессор требует torchvision и
        # обрезает кадр по центру — нам нужен весь кадр, поэтому готовим кадры сами)
        self.mean = np.array([0.485, 0.456, 0.406], np.float32) * 255
        self.std = np.array([0.229, 0.224, 0.225], np.float32) * 255

    def embed(self, clips: np.ndarray, batch: int = 8) -> np.ndarray:
        """[N, VIEWS, FRAMES, S, S, 3] uint8 -> [N, VIEWS, D] float32."""
        torch = self.torch
        n, nv = clips.shape[:2]
        flat = clips.reshape(n * nv, *clips.shape[2:])
        out = []
        for i in range(0, len(flat), batch):
            x = (flat[i:i + batch].astype(np.float32) - self.mean) / self.std  # B, T, H, W, C
            x = torch.from_numpy(x).permute(0, 1, 4, 2, 3).to(self.device, self.dtype)  # B, T, C, H, W
            with torch.inference_mode():
                f = self.model.get_vision_features(pixel_values_videos=x)
            v = torch.nn.functional.normalize(f.float().mean(1), dim=-1)
            out.append(v.cpu().numpy())
        return np.concatenate(out).reshape(n, nv, -1)


@dataclass
class Bank:
    """Размеченные клипы: эмбеддинги [N, VIEWS, D], метка (тип T* или normal), ролик-источник, центр клипа."""
    emb: np.ndarray
    label: np.ndarray
    video: np.ndarray
    t: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "Bank":
        z = np.load(path, allow_pickle=False)
        return cls(z["emb"], z["label"], z["video"], z["t"])

    def save(self, path: Path) -> None:
        np.savez_compressed(path, emb=self.emb.astype(np.float16), label=self.label, video=self.video, t=self.t)


def similarity(q: np.ndarray, emb: np.ndarray) -> np.ndarray:
    """Сходство запроса [VIEWS, D] с базой [N, VIEWS, D]: среднее косинусное по видам -> [N]."""
    return np.einsum("vd,nvd->n", q.astype(np.float32), emb.astype(np.float32)) / q.shape[0]


def scores(q: np.ndarray, bank: Bank, type_id: str, exclude: set[str] = frozenset(), k: int = 3) -> dict:
    """На что похож клип: среднее top-k сходство с примерами типа, с нормой и с любым нарушением.

    exclude — ролики, которые нельзя использовать (сам проверяемый ролик и его дубли при оценке).
    """
    keep = ~np.isin(bank.video, list(exclude)) if exclude else np.ones(len(bank.label), bool)
    sim = similarity(q, bank.emb)

    def topk(mask):
        s = np.sort(sim[mask & keep])[::-1][:k]
        return float(s.mean()) if len(s) else None

    viol = np.char.startswith(bank.label.astype(str), "T")
    best = int(np.argmax(np.where(viol & keep, sim, -9))) if (viol & keep).any() else None
    return {
        "type": topk(bank.label == type_id),
        "normal": topk(bank.label == "normal"),
        "violation": topk(viol),
        "nearest_violation": (str(bank.label[best]), str(bank.video[best]), float(bank.t[best])) if best is not None else None,
    }


def references(q: np.ndarray, bank: Bank, type_id: str, videos_dir: Path, exclude: set[str] = frozenset(),
               n_frames: int = 4) -> list[tuple[str, list[np.ndarray]]]:
    """Ближайший размеченный пример нарушения этого типа и ближайший пример нормы — кадрами из исходных роликов."""
    keep = ~np.isin(bank.video, list(exclude)) if exclude else np.ones(len(bank.label), bool)
    sim = similarity(q, bank.emb)
    out = []
    for label, mask in (("violation", bank.label == type_id), ("normal", bank.label == "normal")):
        mask = mask & keep
        if not mask.any():
            return []  # без пары «нарушение + норма» сравнивать не с чем
        i = int(np.argmax(np.where(mask, sim, -9)))
        c = float(bank.t[i])
        times = list(np.linspace(c - CLIP_SEC / 2 + 0.25, c + CLIP_SEC / 2 - 0.25, n_frames))
        frames = [fr for _, fr in grab(videos_dir / f"{bank.video[i]}.mov", [max(0.0, x) for x in times], long_side=None)]
        if not frames:
            return []
        out.append((label, frames))
    return out
