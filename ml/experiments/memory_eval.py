"""Оценка памяти примеров V-JEPA 2 (без GPU, по эмбеддингам из vjepa_embed.py), строго leave-one-video-out:
при оценке ролика его собственные клипы (и дубли) из базы исключаются.

    python -m ml.experiments.memory_eval --stage1 eval/out/vlm_eval_Qwen3-VL-30B-A3B-Instruct-FP8_final.json \
        --verify eval/out/verify/final.json [--tag vitl] [--views 0 1 2]

A. Разделимость: отличает ли поиск похожих клипы нарушений от клипов нормальных роликов (AUC) и угадывает ли тип.
B. Политики поверх v1: снимать тревогу, если кандидат похож на норму сильнее, чем на примеры своего типа;
   возвращать отклонённых проверкой кандидатов, если они сильно похожи на примеры своего типа; менять тип.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ml.cashier.memory import Bank, scores
from ml.experiments.policy_eval import apply_policy
from ml.experiments.verify_eval import OUT, load_truth, metrics

V1_VERIFY = {t: "verify" for t in ("T1", "T4", "T5", "T8", "T9", "T13")}
EVENT_OFFSETS = (1.0, 2.0, 3.0)  # центры клипов-примеров после метки: действие начинается в момент метки


def load_embeddings(tag: str, views: list[int], center: bool = False) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """center — вычесть средний эмбеддинг ролика: убрать «какая это комната и ракурс», оставить то, что меняется."""
    res = {}
    for f in sorted((OUT / "vjepa" / tag).glob("*.npz")):
        z = np.load(f)
        e = z["emb"][:, views].astype(np.float32)
        if center:
            e = e - e.mean(axis=0, keepdims=True)
            e /= np.linalg.norm(e, axis=-1, keepdims=True) + 1e-8
        res[f.stem] = (z["t"], e)
    return res


def nearest(emb_v, t):
    ts, e = emb_v
    return e[int(np.argmin(np.abs(ts - t)))]


def build_bank(embs, truth) -> Bank:
    rows = []  # (emb, label, video, t)
    for name, events in truth.items():
        if name not in embs:
            continue
        ts, e = embs[name]
        if not events:  # нормальный ролик: все клипы — норма
            rows += [(e[i], "normal", name, ts[i]) for i in range(len(ts))]
            continue
        for t_lab, types in events:
            for off in EVENT_OFFSETS:
                for typ in types:
                    rows.append((nearest(embs[name], t_lab + off), typ, name, t_lab + off))
    return Bank(np.stack([r[0] for r in rows]), np.array([r[1] for r in rows]),
                np.array([r[2] for r in rows]), np.array([r[3] for r in rows], np.float32))


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--verify", required=True)
    ap.add_argument("--tag", default="vitl")
    ap.add_argument("--views", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="без построчного списка кандидатов")
    a = ap.parse_args()

    truth = load_truth()
    embs = load_embeddings(a.tag, a.views, a.center)
    bank = build_bank(embs, truth)
    print(f"bank: {len(bank.label)} clips; " + ", ".join(f"{k}={v}" for k, v in
          sorted(zip(*np.unique(bank.label, return_counts=True)), key=lambda x: x[0])))

    # ---------- A. разделимость
    pos, neg, top1 = [], [], []
    for name, events in truth.items():
        if name not in embs:
            continue
        if events:
            for t_lab, types in events:
                s = scores(nearest(embs[name], t_lab + 2.0), bank, types[0], {name}, a.k)
                if s["violation"] is not None and s["normal"] is not None:
                    pos.append(s["violation"] - s["normal"])
                    top1.append(s["nearest_violation"][0] in types)
        else:
            ts, e = embs[name]
            for i in range(0, len(ts), 4):
                s = scores(e[i], bank, "T1", {name}, a.k)
                neg.append(s["violation"] - s["normal"])
    print(f"\nA. события vs норма: AUC={auc(pos, neg):.3f} (0.5 — не различает); "
          f"тип ближайшего примера верен {sum(top1)}/{len(top1)}")
    print(f"   отрыв (нарушение − норма): события медиана {np.median(pos):+.3f}, нормальные клипы медиана {np.median(neg):+.3f}")

    # ---------- B. политики поверх v1
    stage1 = json.loads(Path(a.stage1).read_text(encoding="utf-8"))
    cache = json.loads(Path(a.verify).read_text(encoding="utf-8"))
    cands = [{"video": r["video"], "type": e["type"], "t": float(e["t"])} for r in stage1["rows"] for e in r["events"]]
    cands = [c for c in cands if c["video"] in embs]
    for c in cands:
        q = nearest(embs[c["video"]], c["t"] + 2.0)
        c["s"] = scores(q, bank, c["type"], {c["video"]}, a.k)
        # лучший тип по памяти (для смены типа)
        by_type = {}
        for typ in sorted(set(bank.label) - {"normal"}):
            v = scores(q, bank, typ, {c["video"]}, a.k)["type"]
            if v is not None:
                by_type[typ] = v
        c["best_type"] = max(by_type, key=by_type.get) if by_type else None
        c["best_score"] = by_type.get(c["best_type"]) if by_type else None
        r = cache.get(f"{c['video']}|{c['type']}|{c['t']:.2f}")
        c["verified"] = bool(r and r["answer"] == "yes")

    def gap(c):  # >0 — больше похож на норму, чем на свой тип; None — примеров типа нет
        s = c["s"]
        return None if s["type"] is None or s["normal"] is None else s["normal"] - s["type"]

    def run(drop_margin=None, rescue_margin=None, retype_margin=None):
        preds = defaultdict(list)
        for c in cands:
            g = gap(c)
            kept = c["verified"] if c["type"] in V1_VERIFY else True
            if not kept and rescue_margin is not None and g is not None and -g > rescue_margin:
                kept = True  # проверка отклонила, но клип очень похож на примеры этого типа
            if kept and drop_margin is not None and g is not None and g > drop_margin:
                kept = False  # похож на норму сильнее, чем на свой тип
            if not kept:
                continue
            typ = c["type"]
            if (retype_margin is not None and c["best_type"] and c["best_type"] != typ and c["s"]["type"] is not None
                    and c["best_score"] - c["s"]["type"] > retype_margin):
                typ = c["best_type"]
            preds[c["video"]].append({"type": typ, "t": c["t"]})
        return metrics(preds, truth)

    def show(name, m):
        print(f"{name:44s} {m['missed']:8d} {m['false_alarms']:6d} {m['verdict_acc']:7.3f} "
              f"{m['localized_3s']:6d} {m['typed_3s']:5d}")

    print(f"\nB. {'политика':41s} пропуски ложные вердикт лок±3 тип±3")
    show("v1 (как сейчас)", metrics(apply_policy(cands, cache, V1_VERIFY), truth))
    for m in (0.0, 0.01, 0.02, 0.03, 0.05):
        show(f"v1 + снять, если норма > тип на {m:.2f}", run(drop_margin=m))
    for m in (0.0, 0.02, 0.05):
        show(f"v1 + вернуть, если тип > норма на {m:.2f}", run(rescue_margin=m))
    for m in (0.0, 0.02, 0.05):
        show(f"v1 + сменить тип при отрыве {m:.2f}", run(retype_margin=m))

    if a.quiet:
        return
    print("\nкандидаты: видео тип t | проверка | тип/норма/нарушение | ближайший пример")
    for c in sorted(cands, key=lambda c: (c["video"], c["t"])):
        s = c["s"]
        f = lambda v: "  —  " if v is None else f"{v:.3f}"  # noqa: E731
        is_tp = any(c["type"] in types and abs(c["t"] - t) <= 3 for t, types in truth.get(c["video"], []))
        print(f"{c['video']:5s} {c['type']:4s} {c['t']:5.1f} {'TP' if is_tp else 'fp'} "
              f"{'yes' if c['verified'] else ('no ' if c['type'] in V1_VERIFY else ' - ')} | "
              f"{f(s['type'])} {f(s['normal'])} {f(s['violation'])} | {s['nearest_violation']}")


if __name__ == "__main__":
    main()
