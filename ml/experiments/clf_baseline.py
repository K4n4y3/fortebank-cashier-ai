"""Базовая проверка подхода «замороженные признаки + multi-label классификатор».

Признаки: SigLIP 2 (google/siglip2-base-patch16-384, Apache-2.0), кадры 2 fps, полный кадр.
Окна 2 с: [среднее эмбеддингов, разность последний-первый].
Голова: StandardScaler -> PCA -> LogisticRegression на каждый тип + отдельная голова «любое нарушение».
Оценка: leave-one-video-out по 40 уникальным роликам (X-47≡X-4, X-48≡X-5 объединены).
Порог вердикта выбирается честно: внутренним LOO на обучающих роликах так, чтобы не было пропусков.

    python ml/experiments/clf_baseline.py extract [--flip]
    python ml/experiments/clf_baseline.py eval [--flip-test]
"""
import argparse
import csv
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VIDEOS = ROOT / "dataset" / "videos"
MODEL_ID = "google/siglip2-base-patch16-384"
CACHE = ROOT / "dataset" / "cache" / "siglip2_384"
OUT = ROOT / "eval" / "out"
FPS = 2.0
DUPLICATES = {"X-47": "X-4", "X-48": "X-5"}  # побайтно одинаковые файлы
EVENT_BEFORE, EVENT_AFTER = 1.0, 3.0  # окно считается положительным в [t-1, t+3] от метки
WIN = 2.0


def load_events() -> dict[str, list[tuple[float, list[str]]]]:
    """video -> [(label_sec, [types])], дубликаты слиты в оригинал."""
    events: dict[str, list] = defaultdict(list)
    with open(ROOT / "eval" / "labels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = DUPLICATES.get(row["video"], row["video"])
            events.setdefault(name, [])
            if row["violation"] == "1":
                events[name].append((float(row["label_sec"]), row["types"].split(";")))
    return dict(events)


# ---------------------------------------------------------------- extract

def extract(flip: bool):
    import cv2
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    CACHE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()

    def embed(images):
        out = []
        for i in range(0, len(images), 16):
            inp = proc(images=images[i:i + 16], return_tensors="pt")
            with torch.inference_mode():
                f = model.get_image_features(**inp)
            f = f.pooler_output if hasattr(f, "pooler_output") else f
            out.append(torch.nn.functional.normalize(f, dim=-1).numpy())
        return np.concatenate(out)

    names = sorted(load_events())
    total_frames, t_all = 0, time.time()
    for name in names:
        path = CACHE / f"{name}.npz"
        if path.exists() and (not flip or "flip" in np.load(path).files):
            continue
        cap = cv2.VideoCapture(str(VIDEOS / f"{name}.mov"))
        src = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, round(src / FPS))
        times, imgs, i = [], [], 0
        while cap.grab():
            if i % step == 0:
                ok, fr = cap.retrieve()
                if ok:
                    times.append(i / src)
                    imgs.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
            i += 1
        cap.release()
        t0 = time.time()
        data = {"times": np.array(times), "emb": embed(imgs)}
        if flip:
            data["flip"] = embed([im.transpose(Image.FLIP_LEFT_RIGHT) for im in imgs])
        np.savez(path, **data)
        total_frames += len(imgs)
        dt = time.time() - t0
        print(f"{name}: {len(imgs)} frames {dt:.1f}s ({dt / len(imgs):.2f}s/frame)", flush=True)
    print(f"done: {total_frames} frames in {time.time() - t_all:.0f}s")


# ---------------------------------------------------------------- eval

def windows(times: np.ndarray, emb: np.ndarray):
    feats, centers = [], []
    for t in times:
        m = (times >= t - WIN / 2) & (times < t + WIN / 2)
        e = emb[m]
        feats.append(np.concatenate([e.mean(0), e[-1] - e[0]]))
        centers.append(t)
    return np.array(feats, dtype=np.float32), np.array(centers)


def window_labels(centers, evs, classes):
    y = np.zeros((len(centers), len(classes)), dtype=np.int8)
    for t_lab, types in evs:
        m = (centers >= t_lab - EVENT_BEFORE) & (centers <= t_lab + EVENT_AFTER)
        for typ in types:
            y[m, classes.index(typ)] = 1
    return y


def make_head(n_pos_dims: int):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        PCA(n_components=n_pos_dims, random_state=0),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000),
    )


def fit_predict(train, test, col, pca_dims):
    """train/test: списки (X, Y); col: индекс столбца Y (или 'any'). Возвращает вероятности для test-окон."""
    X = np.concatenate([x for x, _ in train])
    y = np.concatenate([(yy.max(1) if col == "any" else yy[:, col]) for _, yy in train])
    if y.sum() == 0:
        return None  # класса нет в обучении — предсказать нельзя
    head = make_head(min(pca_dims, X.shape[0] - 1, X.shape[1]))
    head.fit(X, y)
    return [head.predict_proba(x)[:, 1] for x, _ in test]


def evaluate(flip_test: bool, pca_dims: int = 64):
    from sklearn.metrics import roc_auc_score

    events = load_events()
    names = sorted(events)
    classes = sorted({t for evs in events.values() for _, ts in evs for t in ts}, key=lambda s: int(s[1:]))
    data = {}
    for n in names:
        z = np.load(CACHE / f"{n}.npz")
        X, c = windows(z["times"], z["emb"])
        Xf = windows(z["times"], z["flip"])[0] if flip_test else None
        data[n] = dict(X=X, Xf=Xf, centers=c, Y=window_labels(c, events[n], classes))

    rows, t0 = [], time.time()
    for k, held in enumerate(names):
        train_names = [n for n in names if n != held]
        train = [(data[n]["X"], data[n]["Y"]) for n in train_names]
        Xtest = data[held]["Xf"] if flip_test else data[held]["X"]
        test = [(Xtest, data[held]["Y"])]

        p_any = fit_predict(train, test, "any", pca_dims)[0]
        p_cls = {}
        for ci, c in enumerate(classes):
            r = fit_predict(train, test, ci, pca_dims)
            p_cls[c] = None if r is None else r[0]

        # честный порог: внутренний LOO на обучающих роликах, порог = min score среди роликов с нарушением
        inner_scores, inner_pos = [], []
        for inner in train_names:
            tr = [(data[n]["X"], data[n]["Y"]) for n in train_names if n != inner]
            te = [(data[inner]["X"], data[inner]["Y"])]
            s = float(fit_predict(tr, te, "any", pca_dims)[0].max())
            inner_scores.append(s)
            inner_pos.append(bool(events[inner]))
        pos_scores = [s for s, p in zip(inner_scores, inner_pos) if p]
        thr = min(pos_scores) * 0.98 if pos_scores else 0.5

        score = float(p_any.max())
        peak_t = float(data[held]["centers"][int(np.argmax(p_any))])
        ev_rows = []
        for t_lab, types in events[held]:
            near = np.abs(data[held]["centers"] - t_lab) <= 2.0
            best_c, best_p = None, -1.0
            for c in classes:
                if p_cls[c] is not None and near.any():
                    pc = float(p_cls[c][near].max())
                    if pc > best_p:
                        best_c, best_p = c, pc
            seen = [t for t in types if any(t in ts for n in train_names for _, ts in events[n])]
            ev_rows.append(dict(label=t_lab, types=types, pred_type=best_c, type_ok=best_c in types,
                                type_seen_in_train=bool(seen)))
        rows.append(dict(video=held, positive=bool(events[held]), score=round(score, 4), thr=round(thr, 4),
                         pred=score >= thr, pred_05=score >= 0.5, peak=peak_t,
                         dt=min((abs(peak_t - t) for t, _ in events[held]), default=None), events=ev_rows))
        print(f"[{k + 1}/{len(names)}] {held:5s} pos={int(bool(events[held]))} score={score:.3f} thr={thr:.3f} "
              f"pred={int(score >= thr)} peak={peak_t:.1f} " +
              " ".join(f"{e['types']}->{e['pred_type']}{'' if e['type_seen_in_train'] else '(unseen)'}" for e in ev_rows),
              flush=True)

    # ---- сводка
    y = np.array([r["positive"] for r in rows])
    s = np.array([r["score"] for r in rows])
    pred = np.array([r["pred"] for r in rows])
    pred05 = np.array([r["pred_05"] for r in rows])
    auc = roc_auc_score(y, s)
    oracle_thr = s[y].min()
    summary = {
        "videos": len(rows), "positives": int(y.sum()), "negatives": int((~y).sum()),
        "roc_auc": round(float(auc), 3),
        "honest_thr": {"missed": int((y & ~pred).sum()), "false_alarms": int((~y & pred).sum()),
                       "verdict_acc": round(float((y == pred).mean()), 3)},
        "thr_0.5": {"missed": int((y & ~pred05).sum()), "false_alarms": int((~y & pred05).sum()),
                    "verdict_acc": round(float((y == pred05).mean()), 3)},
        "oracle_zero_miss_thr": {"thr": round(float(oracle_thr), 4),
                                 "false_alarms": int((~y & (s >= oracle_thr)).sum())},
    }
    dts = [r["dt"] for r in rows if r["positive"]]
    summary["timecode"] = {"within_2s": sum(d <= 2 for d in dts), "of": len(dts),
                           "median_abs_dt": round(float(np.median(dts)), 1)}
    evs = [e for r in rows for e in r["events"]]
    summary["type"] = {"correct": sum(e["type_ok"] for e in evs), "of": len(evs),
                       "unseen_in_train_fold": sum(not e["type_seen_in_train"] for e in evs),
                       "correct_when_seen": sum(e["type_ok"] for e in evs if e["type_seen_in_train"]),
                       "seen": sum(e["type_seen_in_train"] for e in evs)}
    summary["eval_sec"] = round(time.time() - t0, 1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "flip" if flip_test else "orig"
    (OUT / f"clf_baseline_{tag}.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False,
                                                             indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "eval"])
    ap.add_argument("--flip", action="store_true", help="extract: также признаки зеркальных кадров")
    ap.add_argument("--flip-test", action="store_true", help="eval: тестовый ролик подаётся зеркально (имитация другого ракурса)")
    ap.add_argument("--pca", type=int, default=64)
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.flip)
    else:
        evaluate(a.flip_test, a.pca)
