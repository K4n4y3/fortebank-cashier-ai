"""Оценка второй ступени: проверка кандидатов первой ступени (vlm_eval.py) узкими вопросами.

    python -m ml.experiments.verify_eval --stage1 eval/out/vlm_eval_Qwen3-VL-30B-A3B-Instruct-FP8_v2.json \
        --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 [--videos X-20 X-3]
Ответы проверки кэшируются (eval/out/verify/<tag>.json), повторный запуск только пересчитывает метрики.
"""
import argparse
import concurrent.futures as cf
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ml.cashier.verify import verify

ROOT = Path(__file__).resolve().parents[2]
VIDEOS = ROOT / "dataset" / "videos"
OUT = ROOT / "eval" / "out"
DUPLICATES = {"X-47": "X-4", "X-48": "X-5"}
MATCH = 3.0


def load_truth(include_disputed: bool = False):
    """video -> [(sec, [types])]. Ролики с confidence=disputed (спорная разметка) по умолчанию исключаются."""
    ev = defaultdict(list)
    with open(ROOT / "eval" / "labels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("confidence") == "disputed" and not include_disputed:
                continue
            n = DUPLICATES.get(row["video"], row["video"])
            ev.setdefault(n, [])
            if row["violation"] == "1":
                ev[n].append((float(row["label_sec"]), row["types"].split(";")))
    return dict(ev)


def duration(name):
    cap = cv2.VideoCapture(str(VIDEOS / f"{name}.mov"))
    d = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    cap.release()
    return d


def metrics(preds: dict[str, list[dict]], truth) -> dict:
    """preds: video -> [{type, t}]"""
    y, p, typed, loc, dts = [], [], 0, 0, []
    n_events = 0
    for name, tr in truth.items():
        found = preds.get(name, [])
        y.append(bool(tr))
        p.append(bool(found))
        for t_lab, types in tr:
            n_events += 1
            near = [f for f in found if abs(f["t"] - t_lab) <= MATCH]
            loc += bool(near)
            typed += any(f["type"] in types for f in near)
            if near:
                dts.append(min(abs(f["t"] - t_lab) for f in near))
    y, p = np.array(y), np.array(p)
    return {"missed": int((y & ~p).sum()), "false_alarms": int((~y & p).sum()),
            "verdict_acc": round(float((y == p).mean()), 3), "events": n_events,
            "localized_3s": int(loc), "typed_3s": int(typed), "within_2s": int(sum(d <= 2 for d in dts))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--types", nargs="*", help="проверять только кандидатов этих типов")
    a = ap.parse_args()

    truth = load_truth()
    stage1 = json.loads(Path(a.stage1).read_text(encoding="utf-8"))
    cands = []
    for row in stage1["rows"]:
        if a.videos and row["video"] not in a.videos:
            continue
        for e in row["events"]:
            if a.types and e["type"] not in a.types:
                continue
            cands.append({"video": row["video"], "type": e["type"], "t": float(e["t"]), "obs": e.get("obs", "")})
    if a.videos:
        truth = {k: v for k, v in truth.items() if k in a.videos}

    cache_path = OUT / "verify" / f"{a.tag}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() and not a.force else {}
    durs = {n: duration(n) for n in {c["video"] for c in cands}}

    todo = [c for c in cands if f"{c['video']}|{c['type']}|{c['t']:.2f}" not in cache]
    print(f"candidates: {len(cands)}, to verify: {len(todo)}", flush=True)

    def work(c):
        v = verify(VIDEOS / f"{c['video']}.mov", c["type"], c["t"], durs[c["video"]], a.api, a.model)
        return c, v

    with cf.ThreadPoolExecutor(a.parallel) as ex:
        for c, v in ex.map(work, todo):
            cache[f"{c['video']}|{c['type']}|{c['t']:.2f}"] = {"answer": v.answer, "t_v": v.t, "evidence": v.evidence,
                                                              "focus": v.focus, "fields": v.fields}
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    # --- метрики
    s1, s2, s2t = defaultdict(list), defaultdict(list), defaultdict(list)
    per_type = defaultdict(lambda: {"tp_kept": 0, "tp_dropped": 0, "fp_kept": 0, "fp_dropped": 0})
    for c in cands:
        r = cache[f"{c['video']}|{c['type']}|{c['t']:.2f}"]
        s1[c["video"]].append({"type": c["type"], "t": c["t"]})
        is_tp = any(c["type"] in types and abs(c["t"] - t_lab) <= MATCH for t_lab, types in truth.get(c["video"], []))
        kept = r["answer"] == "yes"
        per_type[c["type"]][("tp_" if is_tp else "fp_") + ("kept" if kept else "dropped")] += 1
        if kept:
            s2[c["video"]].append({"type": c["type"], "t": c["t"]})
            s2t[c["video"]].append({"type": c["type"], "t": r["t_v"]})
        print(f"{c['video']:5s} {c['type']:4s} t={c['t']:5.1f} {'TP' if is_tp else 'fp'} -> {r['answer']:7s} "
              f"[{r['focus']}] {r['evidence'][:110]}")
    summary = {
        "stage1": metrics(s1, truth),
        "stage2_candidate_time": metrics(s2, truth),
        "stage2_verifier_time": metrics(s2t, truth),
        "per_type": dict(sorted(per_type.items())),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=int))
    (OUT / f"verify_eval_{a.tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=int),
                                                   encoding="utf-8")


if __name__ == "__main__":
    main()
