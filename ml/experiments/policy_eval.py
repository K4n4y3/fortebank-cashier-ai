"""Сравнение политик второй ступени по кэшу (без GPU): для каждого типа keep / verify / drop.

    python -m ml.experiments.policy_eval --stage1 eval/out/vlm_eval_Qwen3-VL-30B-A3B-Instruct-FP8_v2.json \
        --verify eval/out/verify/full1.json [--verify-override T1=eval/out/verify/t1full.json]
"""
import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

from ml.experiments.verify_eval import load_truth, metrics

TYPES = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12"]


def apply_policy(cands, cache, policy, cache_by_type=None):
    preds = defaultdict(list)
    for c in cands:
        mode = policy.get(c["type"], "keep")
        if mode == "drop":
            continue
        if mode == "verify":
            src = (cache_by_type or {}).get(c["type"], cache)
            r = src.get(f"{c['video']}|{c['type']}|{c['t']:.2f}")
            if not r or r["answer"] != "yes":
                continue
        preds[c["video"]].append({"type": c["type"], "t": c["t"]})
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--verify", required=True)
    ap.add_argument("--verify-override", nargs="*", default=[], help="TYPE=path: другой кэш проверки для типа")
    a = ap.parse_args()

    truth = load_truth()
    stage1 = json.loads(Path(a.stage1).read_text(encoding="utf-8"))
    cands = [{"video": r["video"], "type": e["type"], "t": float(e["t"])} for r in stage1["rows"] for e in r["events"]]
    cache = json.loads(Path(a.verify).read_text(encoding="utf-8"))
    by_type = {}
    for ov in a.verify_override:
        typ, path = ov.split("=", 1)
        by_type[typ] = json.loads(Path(path).read_text(encoding="utf-8"))

    named = {
        "stage1 (всё принимать)": {},
        "проверять всё": {t: "verify" for t in TYPES},
        "проверять T1,T4,T8; T9 принимать": {"T1": "verify", "T4": "verify", "T8": "verify"},
        "проверять T1,T4,T8,T9": {"T1": "verify", "T4": "verify", "T8": "verify", "T9": "verify"},
        "проверять T1,T4,T8; T9 отбросить": {"T1": "verify", "T4": "verify", "T8": "verify", "T9": "drop"},
        "проверять T4,T8; T1 принимать; T9 проверять": {"T4": "verify", "T8": "verify", "T9": "verify"},
    }
    print(f"{'политика':48s} пропуски ложные вердикт лок±3 тип±3 ±2с")
    for name, pol in named.items():
        m = metrics(apply_policy(cands, cache, pol, by_type), truth)
        print(f"{name:48s} {m['missed']:8d} {m['false_alarms']:6d} {m['verdict_acc']:7.3f} {m['localized_3s']:6d} "
              f"{m['typed_3s']:5d} {m['within_2s']:4d}")

    # перебор: лучшая политика при нуле/минимуме пропусков (осторожно: подгонка под 40 роликов)
    present = sorted({c["type"] for c in cands}, key=lambda s: int(s[1:]) if s[1:].isdigit() else 99)
    best = []
    for combo in itertools.product(["keep", "verify", "drop"], repeat=len(present)):
        pol = dict(zip(present, combo))
        m = metrics(apply_policy(cands, cache, pol, by_type), truth)
        best.append(((m["missed"], m["false_alarms"], -m["typed_3s"]), pol, m))
    best.sort(key=lambda x: x[0])
    print("\nлучшие по (пропуски, ложные, -тип):")
    for key, pol, m in best[:5]:
        print(key, {k: v for k, v in pol.items() if v != "keep"}, m)


if __name__ == "__main__":
    main()
