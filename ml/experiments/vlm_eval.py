"""Zero-shot проверка VLM (Qwen3-VL / Qwen2.5-VL через vLLM) на размеченных роликах.

Ролик режется на перекрывающиеся окна; каждое окно — набор кадров с подписями времени.
VLM получает чек-лист типов T1–T12 + OTHER и отвечает строгим JSON. События окон сливаются,
вердикт на ролик = есть событие с confidence >= порога. Метрики совпадают с clf_baseline.py.

На GPU-сервере (vLLM уже запущен на 127.0.0.1:8000):
    python -m ml.experiments.vlm_eval --model Qwen/Qwen3-VL-8B-Instruct
    python -m ml.experiments.vlm_eval --model ... --videos X-2 X-20 --dump   # отладка ответов
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ml.cashier.detect import detect  # единый код первой ступени с продуктом

ROOT = Path(__file__).resolve().parents[2]
VIDEOS = ROOT / "dataset" / "videos"
OUT = ROOT / "eval" / "out"
DUPLICATES = {"X-47": "X-4", "X-48": "X-5"}

def load_events():
    ev = defaultdict(list)
    with open(ROOT / "eval" / "labels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n = DUPLICATES.get(row["video"], row["video"])
            ev.setdefault(n, [])
            if row["violation"] == "1":
                ev[n].append((float(row["label_sec"]), row["types"].split(";")))
    return dict(ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--stride", type=float, default=5.0)
    ap.add_argument("--long-side", type=int, default=848)
    ap.add_argument("--tag", default="")
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--dump", action="store_true")
    a = ap.parse_args()

    events = load_events()
    names = a.videos or sorted(events)
    rows = []
    for name in names:
        t_start = time.time()
        merged, raw = detect(VIDEOS / f"{name}.mov", a.api, a.model, a.fps, a.window, a.stride, a.long_side,
                             a.parallel, a.thr)
        truth = events[name]
        pred = bool(merged)
        dt = None
        if truth and merged:
            dt = min(abs(e["t"] - t) for e in merged for t, _ in truth)
        type_ok = [any(e["type"] in types and abs(e["t"] - t) <= 3 for e in merged) for t, types in truth]
        rows.append(dict(video=name, positive=bool(truth), pred=pred, dt=dt, type_ok=type_ok,
                         events=merged, sec=round(time.time() - t_start, 1)))
        print(f"{name:5s} pos={int(bool(truth))} pred={int(pred)} dt={dt} types={[ts for _, ts in truth]} "
              f"found={[(e['type'], round(e['t'], 1), round(e['conf'], 2)) for e in merged]} {rows[-1]['sec']}s",
              flush=True)
        if a.dump:
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / f"vlm_raw_{name}.json").write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")

    y = np.array([r["positive"] for r in rows])
    p = np.array([r["pred"] for r in rows])
    dts = [r["dt"] for r in rows if r["positive"] and r["dt"] is not None]
    tok = [x for r in rows for x in r["type_ok"]]
    summary = {
        "model": a.model, "videos": len(rows),
        "missed": int((y & ~p).sum()), "false_alarms": int((~y & p).sum()),
        "verdict_acc": round(float((y == p).mean()), 3),
        "timecode_within_2s": sum(d <= 2 for d in dts), "of_detected": len(dts),
        "type_correct": sum(tok), "type_of": len(tok),
        "sec_per_video_mean": round(float(np.mean([r["sec"] for r in rows])), 1),
        "params": vars(a),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    tag = a.model.split("/")[-1] + (f"_{a.tag}" if a.tag else "")
    (OUT / f"vlm_eval_{tag}.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False,
                                                         indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
