"""Оценка схемы «VLM-восприятие + правила» на размеченных роликах.

Шаг 1 (GPU, vLLM): восприятие всех роликов -> кэш eval/out/timeline/<model>/<video>.json
Шаг 2 (где угодно): правила по кэшу -> метрики. Правила можно менять и пересчитывать без VLM.

    python -m ml.experiments.timeline_eval --model Qwen/Qwen3-VL-8B-Instruct          # восприятие + оценка
    python -m ml.experiments.timeline_eval --model Qwen/Qwen3-VL-8B-Instruct --rules-only
"""
import argparse
import concurrent.futures as cf
import csv
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ml.cashier import rules
from ml.cashier.perception import Timeline, perceive, perceive_frames
from ml.cashier.video import sample

ROOT = Path(__file__).resolve().parents[2]
VIDEOS = ROOT / "dataset" / "videos"
OUT = ROOT / "eval" / "out"
DUPLICATES = {"X-47": "X-4", "X-48": "X-5"}
MATCH_SEC = 3.0  # событие найдено, если предсказание того же ролика в пределах ±3 с от метки


def load_events():
    ev = defaultdict(list)
    with open(ROOT / "eval" / "labels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n = DUPLICATES.get(row["video"], row["video"])
            ev.setdefault(n, [])
            if row["violation"] == "1":
                ev[n].append((float(row["label_sec"]), row["types"].split(";")))
    return dict(ev)


def run_perception(name: str, a, cache_dir: Path) -> tuple[str, float]:
    path = cache_dir / f"{name}.json"
    if path.exists() and not a.force:
        return name, 0.0
    t0 = time.time()
    frames, dur = sample(VIDEOS / f"{name}.mov", a.fps, a.long_side)
    if a.mode == "frames":
        tl = perceive_frames(frames, a.api, a.model, a.parallel)
    else:
        tl = perceive(frames, a.api, a.model, a.window, a.stride, a.parallel)
    d = tl.to_json()
    d["duration"] = dur
    d["perception_sec"] = round(time.time() - t0, 1)
    path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return name, d["perception_sec"]


def evaluate(cache_dir: Path, names: list[str], params: rules.Params, verbose=True) -> dict:
    events = load_events()
    rows = []
    per_type = defaultdict(lambda: {"hit": 0, "n": 0})
    for name in names:
        d = json.loads((cache_dir / f"{name}.json").read_text(encoding="utf-8"))
        d.pop("duration", None)
        sec = d.pop("perception_sec", None)
        found = rules.apply(Timeline.from_json(d), params)
        truth = events[name]
        ev_rows = []
        for t_lab, types in truth:
            near = [f for f in found if abs(f.t - t_lab) <= MATCH_SEC]
            typed = [f for f in near if f.type_id in types]
            for typ in types:
                per_type[typ]["n"] += 1
                per_type[typ]["hit"] += int(any(f.type_id == typ for f in near))
            ev_rows.append(dict(label=t_lab, types=types, localized=bool(near), typed=bool(typed),
                                dt=min((abs(f.t - t_lab) for f in near), default=None)))
        rows.append(dict(video=name, positive=bool(truth), pred=bool(found), sec=sec, events=ev_rows,
                         found=[asdict(f) for f in found]))
        if verbose:
            print(f"{name:5s} pos={int(bool(truth))} pred={int(bool(found))} truth={[(t, ts) for t, ts in truth]} "
                  f"found={[(f.type_id, round(f.t, 1)) for f in found]}")
    y = np.array([r["positive"] for r in rows])
    p = np.array([r["pred"] for r in rows])
    evs = [e for r in rows for e in r["events"]]
    secs = [r["sec"] for r in rows if r["sec"]]
    summary = {
        "videos": len(rows),
        "missed_videos": int((y & ~p).sum()), "false_alarm_videos": int((~y & p).sum()),
        "verdict_acc": round(float((y == p).mean()), 3),
        "events_localized_3s": sum(e["localized"] for e in evs), "events_typed_3s": sum(e["typed"] for e in evs),
        "events": len(evs),
        "per_type_recall": {k: f"{v['hit']}/{v['n']}" for k, v in sorted(per_type.items(), key=lambda kv: int(kv[0][1:]))},
        "false_findings_on_normal": sum(len(r["found"]) for r in rows if not r["positive"]),
        "perception_sec_mean": round(float(np.mean(secs)), 1) if secs else None,
        "perception_sec_max": round(float(np.max(secs)), 1) if secs else None,
    }
    return {"summary": summary, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--stride", type=float, default=5.0)
    ap.add_argument("--long-side", type=int, default=848)
    ap.add_argument("--parallel", type=int, default=64, help="запросов одного ролика параллельно")
    ap.add_argument("--mode", choices=["frames", "windows"], default="frames")
    ap.add_argument("--videos-parallel", type=int, default=3)
    ap.add_argument("--rules-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--use-other", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    tag = a.model.split("/")[-1] + (f"_{a.tag}" if a.tag else "")
    cache_dir = OUT / "timeline" / tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    names = a.videos or sorted(load_events())

    if not a.rules_only:
        t0 = time.time()
        with cf.ThreadPoolExecutor(a.videos_parallel) as ex:
            for name, sec in ex.map(lambda n: run_perception(n, a, cache_dir), names):
                print(f"perceived {name}: {sec}s", flush=True)
        print(f"perception total {time.time() - t0:.0f}s")

    res = evaluate(cache_dir, names, rules.Params(use_other=a.use_other))
    print(json.dumps(res["summary"], ensure_ascii=False, indent=1))
    (OUT / f"timeline_eval_{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                                                   encoding="utf-8")


if __name__ == "__main__":
    main()
