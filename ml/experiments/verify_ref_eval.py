"""v2: повторная проверка кандидатов с размеченными примерами из памяти (leave-one-video-out).

Для каждого кандидата первой ступени берётся ближайший пример нарушения того же типа и ближайший пример нормы
из других роликов (по эмбеддингам V-JEPA 2), их кадры показываются модели перед проверяемыми кадрами.
Если примеров типа в других роликах нет, кандидат проверяется как в v1 (ответ берётся из кэша v1).

    python -m ml.experiments.verify_ref_eval --stage1 eval/out/vlm_eval_Qwen3-VL-30B-A3B-Instruct-FP8_final.json \
        --v1-cache eval/out/verify/final.json --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 [--tag ref1]
Кэш ответов: eval/out/verify/<tag>.json (тот же формат, что у verify_eval.py, плюс поле refs).
"""
import argparse
import concurrent.futures as cf
import json
from pathlib import Path

from ml.cashier import memory
from ml.cashier.verify import verify
from ml.experiments.memory_eval import build_bank, load_embeddings, nearest
from ml.experiments.policy_eval import apply_policy
from ml.experiments.verify_eval import OUT, VIDEOS, duration, load_truth, metrics

V1_VERIFY = {t: "verify" for t in ("T1", "T4", "T5", "T8", "T9", "T13")}


def verify_where_possible(cands, cache):
    """Кандидат проходит, если проверка с примерами сказала yes; без примеров — по правилу v1."""
    preds = {}
    for c in cands:
        r = cache.get(f"{c['video']}|{c['type']}|{c['t']:.2f}")
        if r is not None:
            ok = r["answer"] == "yes"
        else:
            ok = c["type"] not in V1_VERIFY  # примеров нет и в v1 тип не проверялся — принимаем, как v1
        if ok:
            preds.setdefault(c["video"], []).append({"type": c["type"], "t": c["t"]})
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--v1-cache", required=True)
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="ref1")
    ap.add_argument("--emb-tag", default="vitl")
    ap.add_argument("--parallel", type=int, default=8)
    a = ap.parse_args()

    truth = load_truth()
    embs = load_embeddings(a.emb_tag, [0, 1, 2])
    bank = build_bank(embs, truth)
    stage1 = json.loads(Path(a.stage1).read_text(encoding="utf-8"))
    v1 = json.loads(Path(a.v1_cache).read_text(encoding="utf-8"))
    cands = [{"video": r["video"], "type": e["type"], "t": float(e["t"])} for r in stage1["rows"] for e in r["events"]]
    cands = [c for c in cands if c["video"] in embs]

    cache_path = OUT / "verify" / f"{a.tag}.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    durs = {n: duration(n) for n in {c["video"] for c in cands}}
    key = lambda c: f"{c['video']}|{c['type']}|{c['t']:.2f}"  # noqa: E731
    todo = [c for c in cands if key(c) not in cache]
    print(f"candidates: {len(cands)}, to verify: {len(todo)}", flush=True)

    def work(c):
        q = nearest(embs[c["video"]], c["t"] + 2.0)
        refs = memory.references(q, bank, c["type"], VIDEOS, exclude={c["video"]})
        if not refs:
            r = v1.get(key(c))
            return c, (dict(r, refs="none (v1 answer)") if r else None)
        v = verify(VIDEOS / f"{c['video']}.mov", c["type"], c["t"], durs[c["video"]], a.api, a.model, references=refs)
        return c, {"answer": v.answer, "t_v": v.t, "evidence": v.evidence, "focus": v.focus, "fields": v.fields,
                   "refs": "violation+normal"}

    with cf.ThreadPoolExecutor(a.parallel) as ex:
        for c, r in ex.map(work, todo):
            if r is not None:
                cache[key(c)] = r
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{'политика':52s} пропуски ложные вердикт лок±3 тип±3")
    rows = {
        "v1: проверка T1,T4,T5,T8,T9,T13 без примеров": apply_policy(cands, v1, V1_VERIFY),
        "v2: те же типы, проверка с примерами": apply_policy(cands, cache, V1_VERIFY),
        "v2: все типы, где есть примеры (остальные как в v1)": verify_where_possible(cands, cache),
    }
    for name, preds in rows.items():
        m = metrics(preds, truth)
        print(f"{name:52s} {m['missed']:8d} {m['false_alarms']:6d} {m['verdict_acc']:7.3f} "
              f"{m['localized_3s']:6d} {m['typed_3s']:5d}")
    for c in sorted(cands, key=lambda c: (c["video"], c["t"])):
        r1, r2 = v1.get(key(c), {}), cache.get(key(c), {})
        is_tp = any(c["type"] in types and abs(c["t"] - t) <= 3 for t, types in truth.get(c["video"], []))
        print(f"{c['video']:5s} {c['type']:4s} {c['t']:5.1f} {'TP' if is_tp else 'fp'} v1={r1.get('answer', '-'):7s} "
              f"v2={r2.get('answer', '-'):7s} {r2.get('refs', '')}")


if __name__ == "__main__":
    main()
