"""Тест EmbeddingGemma 2 на кропах: различает ли модель мелкие атрибуты, если ей показать нужную область.

python ml/spikes/eg2_crops.py --save   — только сохранить кропы для визуальной проверки
python ml/spikes/eg2_crops.py          — закодировать кропы и сделать zero-shot классификацию
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VIDEOS = ROOT / "dataset" / "videos"
OUT = ROOT / "eval" / "out" / "crops"

# (видео, сек, бокс x1,y1,x2,y2 в долях кадра, группа, истинный класс)
CROPS = [
    ("X-20", 3.0, (0.25, 0.22, 0.70, 0.48), "safe", "safe_open"),
    ("X-19", 39.1, (0.45, 0.30, 0.95, 0.60), "safe", "safe_open"),
    ("X-19", 30.2, (0.45, 0.30, 0.95, 0.60), "safe", "safe_closed"),
    ("X-3", 5.0, (0.45, 0.28, 0.95, 0.55), "safe", "safe_closed"),
    ("X-20", 6.5, (0.15, 0.28, 0.75, 0.55), "hands", "phone"),
    ("X-20", 7.5, (0.15, 0.28, 0.75, 0.55), "hands", "phone"),
    ("X-3", 9.0, (0.25, 0.30, 0.85, 0.55), "hands", "id_card"),
    ("X-43", 19.0, (0.10, 0.30, 0.65, 0.52), "hands", "counting"),
    ("X-45", 17.0, (0.10, 0.30, 0.65, 0.52), "hands", "counting"),
    ("X-46", 13.5, (0.10, 0.30, 0.65, 0.52), "hands", "drink"),
    ("X-46", 11.0, (0.10, 0.30, 0.65, 0.52), "hands", "drink"),
    ("X-40", 18.5, (0.05, 0.28, 0.70, 0.55), "hands", "bag"),
]

CLASSES = {
    "safe": {
        "safe_open": "a metal safe with its door standing open",
        "safe_closed": "a metal safe with its door shut",
    },
    "hands": {
        "phone": "a woman looking at a mobile phone in her hands",
        "id_card": "a woman holding a small plastic identity card",
        "counting": "a woman counting a stack of banknotes",
        "drink": "a woman drinking from a paper cup",
        "bag": "a handbag lying on a desk",
    },
}


def grab(name: str, t: float, box):
    cap = cv2.VideoCapture(str(VIDEOS / f"{name}.mov"))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, fr = cap.read()
    cap.release()
    h, w = fr.shape[:2]
    x1, y1, x2, y2 = box
    return fr[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    crops = [grab(n, t, b) for n, t, b, _, _ in CROPS]

    if args.save:
        tiles = []
        for (n, t, _, _, y), c in zip(CROPS, crops):
            c = cv2.resize(c, (300, int(c.shape[0] * 300 / c.shape[1])))
            c = cv2.copyMakeBorder(c, 0, max(0, 200 - c.shape[0]), 0, 0, cv2.BORDER_CONSTANT)[:200]
            cv2.putText(c, f"{n} {t} {y}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            tiles.append(c)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        cv2.imwrite(str(OUT / "crops_sheet.jpg"), sheet)
        print(OUT / "crops_sheet.jpg")
        return

    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("google/embeddinggemma-2", device="cpu",
                                model_kwargs={"torch_dtype": torch.float32},
                                config_kwargs={"audio_config": None})
    imgs = [{"image": Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))} for c in crops]
    iemb = model.encode(imgs, batch_size=4, normalize_embeddings=True)
    correct = 0
    for (n, t, _, g, y), e in zip(CROPS, iemb):
        labels = list(CLASSES[g])
        temb = model.encode([CLASSES[g][k] for k in labels], prompt_name="SearchQuery", normalize_embeddings=True)
        s = temb @ e
        order = np.argsort(-s)
        pred = labels[order[0]]
        correct += pred == y
        ranking = ", ".join(f"{labels[i]}={s[i]:.3f}" for i in order)
        print(f"{n:5s} {t:5.1f} true={y:12s} pred={pred:12s} {'OK ' if pred == y else 'BAD'} | {ranking}")
    print(f"accuracy {correct}/{len(CROPS)}")


if __name__ == "__main__":
    main()
