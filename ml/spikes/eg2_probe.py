"""Проверка EmbeddingGemma 2: дают ли эмбеддинги кадров сигнал о нарушениях.

1) Кодирует кадры роликов (1 fps) и текстовые запросы, кэширует эмбеддинги.
2) Для каждого запроса строит кривую сходства по времени, z-нормирует внутри ролика.
3) Сравнивает пики с разметкой (eval/labels.csv) и с нормальными роликами.

Запуск: .venv\\Scripts\\python ml\\spikes\\eg2_probe.py --videos X-20 X-46 X-3 ...  (или --all)
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
VIDEOS = ROOT / "dataset" / "videos"
CACHE = ROOT / "dataset" / "cache" / "eg2"
OUT = ROOT / "eval" / "out"

PROMPTS = {
    "phone": "a person looking at a smartphone held in their hands",
    "drink": "a person drinking from a cup",
    "safe_open": "a grey metal safe with its door open",
    "safe_closed": "a closed grey metal safe",
    "door_open": "an open glass door of a small office",
    "empty_room": "an empty office room with nobody inside",
    "two_visitors": "two visitors in a small office with a cashier",
    "cash_bag": "a green cash collection bag on a desk",
    "bag_desk": "a handbag and a coat lying on an office desk",
    "money_desk": "banknotes lying on a desk",
    "id_card": "a client handing an identity card to a cashier",
    "counting": "a cashier counting banknotes by hand",
    "leaving": "a woman walking out of the room through the door",
    "entering": "a man entering the room through the door",
    "hand_cash": "a cashier handing money to a client across the desk",
    "hide_cash": "a woman hiding a banknote in her sleeve",
}

# какие запросы должны «загореться» для типа нарушения
TYPE_PROMPTS = {
    "T1": ["empty_room", "leaving", "safe_open"],
    "T2": ["two_visitors", "entering"],
    "T3": ["door_open"],
    "T4": ["phone"],
    "T5": ["drink"],
    "T6": ["bag_desk"],
    "T7": ["cash_bag"],
    "T8": ["hide_cash", "counting"],
    "T9": ["id_card"],
    "T10": ["leaving"],
    "T11": ["money_desk"],
    "T12": ["hand_cash"],
}


def load_labels() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    with open(ROOT / "eval" / "labels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["video"], []).append(row)
    return out


def frames_1fps(path: Path, fps: float) -> list[tuple[float, Image.Image]]:
    cap = cv2.VideoCapture(str(path))
    src = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(src / fps))
    res, i = [], 0
    while cap.grab():
        if i % step == 0:
            ok, fr = cap.retrieve()
            if ok:
                res.append((i / src, Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))))
        i += 1
    cap.release()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    labels = load_labels()
    names = sorted(labels) if args.all else args.videos
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    t0 = time.time()
    model = SentenceTransformer(
        "google/embeddinggemma-2",
        device="cpu",
        model_kwargs={"torch_dtype": torch.float32},
        config_kwargs={"audio_config": None},
    )
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    keys = list(PROMPTS)
    temb = model.encode([PROMPTS[k] for k in keys], prompt_name="SearchQuery", normalize_embeddings=True)

    per_video = {}
    for name in names:
        cache = CACHE / f"{name}_{args.fps:g}fps.npz"
        if cache.exists():
            z = np.load(cache)
            times, femb = z["times"], z["emb"]
        else:
            frames = frames_1fps(VIDEOS / f"{name}.mov", args.fps)
            t1 = time.time()
            femb = model.encode(
                [{"image": im} for _, im in frames],
                batch_size=args.batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            dt = time.time() - t1
            times = np.array([t for t, _ in frames])
            np.savez(cache, times=times, emb=femb)
            print(f"{name}: {len(frames)} frames in {dt:.1f}s ({dt / len(frames):.2f}s/frame)", flush=True)
        sim = femb @ temb.T  # [T, P]
        per_video[name] = {"times": times, "sim": sim}

    # --- анализ ---
    report = {"prompts": keys, "videos": {}}
    normal = [n for n in names if labels[n][0]["violation"] == "0"]
    # максимум «сырого» сходства по каждому запросу в нормальных роликах
    normal_max = {k: [float(per_video[n]["sim"][:, j].max()) for n in normal] for j, k in enumerate(keys)}

    print("\n=== Localization / discrimination per labeled event ===")
    for name in names:
        times, sim = per_video[name]["times"], per_video[name]["sim"]
        zs = (sim - sim.mean(0)) / (sim.std(0) + 1e-6)
        vrep = {"events": []}
        for row in labels[name]:
            if row["violation"] != "1":
                continue
            label_t = float(row["label_sec"])
            for typ in row["types"].split(";"):
                for k in TYPE_PROMPTS.get(typ, []):
                    j = keys.index(k)
                    peak_t = float(times[int(np.argmax(zs[:, j]))])
                    # значение в окне ±2 с от метки
                    win = np.abs(times - label_t) <= 2.0
                    raw_at = float(sim[win, j].max()) if win.any() else float("nan")
                    z_at = float(zs[win, j].max()) if win.any() else float("nan")
                    nm = normal_max[k]
                    pct = float(np.mean([raw_at > x for x in nm])) if nm else float("nan")
                    ev = dict(type=typ, prompt=k, label=label_t, peak=peak_t, dt=round(peak_t - label_t, 1),
                              raw_at_label=round(raw_at, 4), z_at_label=round(z_at, 2),
                              beats_normal_max=round(pct, 2))
                    vrep["events"].append(ev)
                    print(f"{name:5s} {typ:4s} {k:12s} label={label_t:5.1f} peak={peak_t:5.1f} "
                          f"dt={peak_t - label_t:+5.1f} z@label={z_at:5.2f} raw@label={raw_at:.4f} "
                          f">normal_max:{pct:.0%}")
        vrep["max_raw"] = {k: round(float(sim[:, j].max()), 4) for j, k in enumerate(keys)}
        report["videos"][name] = vrep

    print("\n=== Prompt max (raw) in normal videos: median / max ===")
    for k in keys:
        v = normal_max[k]
        if v:
            print(f"{k:12s} median={np.median(v):.4f} max={np.max(v):.4f}")
    (OUT / "eg2_probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez(OUT / "eg2_curves.npz", **{f"{n}__times": per_video[n]["times"] for n in names},
             **{f"{n}__sim": per_video[n]["sim"] for n in names})


if __name__ == "__main__":
    main()
