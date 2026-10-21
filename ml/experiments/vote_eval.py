"""Голосование повторной проверки: решение по большинству из k независимых запусков против одного запуска.

    python -m ml.experiments.vote_eval --stage1 eval/out/vlm_eval_Qwen3-VL-30B-A3B-Instruct-FP8_final.json \
        --runs final final_r2 t0_r3 t0_r4 t0_r5 [--k 3]
Каждый запуск — кэш verify_eval.py (eval/out/verify/<run>.json) по тем же кандидатам первой ступени.
Для честного сравнения считаются все сочетания k запусков из имеющихся: среднее, минимум и максимум.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from ml.experiments.verify_eval import OUT, load_truth, metrics

V1_TYPES = ("T1", "T4", "T5", "T8", "T9", "T13")


def predictions(cands, votes, need):
    """votes: для каждого кандидата число ответов yes; need — сколько yes нужно для подтверждения."""
    preds = {}
    for c in cands:
        if c["type"] in V1_TYPES and votes[c["key"]] < need:
            continue
        preds.setdefault(c["video"], []).append({"type": c["type"], "t": c["t"]})
    return preds


def summary(rows):
    keys = ("missed", "false_alarms", "verdict_acc", "localized_3s", "typed_3s")
    return {k: (float(np.mean([r[k] for r in rows])), min(r[k] for r in rows), max(r[k] for r in rows)) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()

    truth = load_truth()
    stage1 = json.loads(Path(a.stage1).read_text(encoding="utf-8"))
    cands = [{"video": r["video"], "type": e["type"], "t": float(e["t"])} for r in stage1["rows"] for e in r["events"]]
    for c in cands:
        c["key"] = f"{c['video']}|{c['type']}|{c['t']:.2f}"
    runs = {}
    for name in a.runs:
        cache = json.loads((OUT / "verify" / f"{name}.json").read_text(encoding="utf-8"))
        runs[name] = {c["key"]: int(cache.get(c["key"], {}).get("answer") == "yes") for c in cands}

    print(f"{'запуск':28s} пропуски ложные вердикт лок±3 тип±3")
    single = []
    for name, yes in runs.items():
        m = metrics(predictions(cands, yes, 1), truth)
        single.append(m)
        print(f"{name:28s} {m['missed']:8d} {m['false_alarms']:6d} {m['verdict_acc']:7.3f} "
              f"{m['localized_3s']:6d} {m['typed_3s']:5d}")

    checked = [c for c in cands if c["type"] in V1_TYPES]
    flips = sum(len({runs[n][c["key"]] for n in runs}) > 1 for c in checked)
    print(f"\nкандидатов на проверке: {len(checked)}; ответ менялся между запусками у {flips}")

    rules = {"один запуск": None, f"большинство из {a.k}": a.k // 2 + 1,
             f"хотя бы 1 yes из {a.k}": 1, f"все {a.k} yes": a.k}
    print(f"\n{'правило':28s} {'пропуски':>16s} {'ложные':>16s} {'вердикт':>20s} {'тип±3':>14s}   (среднее [мин–макс])")
    for name, need in rules.items():
        if need is None:
            rows = single
        else:
            rows = []
            for combo in itertools.combinations(runs, a.k):
                votes = {c["key"]: sum(runs[n][c["key"]] for n in combo) for c in cands}
                rows.append(metrics(predictions(cands, votes, need), truth))
        s = summary(rows)
        fmt = lambda k, p=1: f"{s[k][0]:.{p}f} [{s[k][1]}–{s[k][2]}]"  # noqa: E731
        print(f"{name:28s} {fmt('missed'):>16s} {fmt('false_alarms'):>16s} {fmt('verdict_acc', 3):>20s} "
              f"{fmt('typed_3s'):>14s}   n={len(rows)}")


if __name__ == "__main__":
    main()
