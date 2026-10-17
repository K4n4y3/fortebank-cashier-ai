"""Нагрузочный тест: N роликов одновременно через продуктовую функцию analyze() (как будто жюри прислало разом).

    python -m ml.experiments.load_test --videos X-14 X-39 X-4 X-5 X-13
Печатает время каждого ролика от «загрузки» до готового результата; лимит жюри — 300 с на ролик.
"""
import argparse
import concurrent.futures as cf
import json
import time

from ml.cashier.pipeline import Config, analyze
from ml.experiments.verify_eval import VIDEOS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=Config.api)
    ap.add_argument("--model", default=Config.model)
    ap.add_argument("--videos", nargs="+", default=["X-14", "X-39", "X-4", "X-5", "X-13"])
    a = ap.parse_args()
    cfg = Config(api=a.api, model=a.model)
    t0 = time.time()

    def run(name):
        r = analyze(VIDEOS / f"{name}.mov", cfg)
        return dict(video=name, done_after=round(time.time() - t0, 1), verdict=r["verdict"],
                    violations=[(d["type_id"], v["timecode_sec"]) for d, v in zip(r["details"], r["violations"])],
                    timing=r["timing"])

    with cf.ThreadPoolExecutor(len(a.videos)) as ex:
        res = list(ex.map(run, a.videos))
    for r in res:
        print(json.dumps(r, ensure_ascii=False))
    worst = max(r["done_after"] for r in res)
    print(f"wall={time.time() - t0:.1f}s worst_video={worst}s {'OK' if worst < 300 else 'OVER 300s LIMIT'}")


if __name__ == "__main__":
    main()
