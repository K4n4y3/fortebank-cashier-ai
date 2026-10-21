"""Эмбеддинги V-JEPA 2 для всех клипов датасета (шаг 1 с) — основа памяти примеров v2.

    python -m ml.experiments.vjepa_embed [--videos X-4 X-20] [--stride 1.0]
Результат: eval/out/vjepa/<tag>/<video>.npz с полями t (центры клипов) и emb [N, VIEWS, D] (float16).
"""
import argparse
import time
from pathlib import Path

import numpy as np

from ml.cashier import memory
from ml.experiments.verify_eval import DUPLICATES, OUT, VIDEOS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=memory.MODEL)
    ap.add_argument("--tag", default="vitl")
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--stride", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out = OUT / "vjepa" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    names = a.videos or sorted(p.stem for p in VIDEOS.glob("*.mov") if p.stem not in DUPLICATES)
    enc = memory.Encoder(a.model)
    t_all = time.time()
    for name in names:
        dst = out / f"{name}.npz"
        if dst.exists() and not a.force:
            continue
        t0 = time.time()
        frames = memory.sampled_views(VIDEOS / f"{name}.mov")
        centers = memory.clip_centers(len(frames), a.stride)
        embs = []
        for i in range(0, len(centers), 16):  # по 16 клипов, чтобы не держать все клипы в памяти
            clips = np.stack([memory.clip_from(frames, c) for c in centers[i:i + 16]])
            embs.append(enc.embed(clips, a.batch))
        emb = np.concatenate(embs)
        np.savez_compressed(dst, t=centers, emb=emb.astype(np.float16))
        print(f"{name}: {len(centers)} clips, emb {emb.shape}, {time.time() - t0:.1f}s", flush=True)
    print(f"done in {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
