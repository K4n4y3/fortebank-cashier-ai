"""Печать покадровой временной линии восприятия: python -m ml.tools.show_timeline <cache_dir> X-22 X-20"""
import json
import sys
from pathlib import Path

KEYS = [("cashier_in_room", "room"), ("cashier_at_workplace", "work"), ("visitors", "vis"), ("safe_open", "safe"),
        ("door_open", "door"), ("cash_on_desk", "cash"), ("cashier_hands", "hands"), ("personal_items_on_desk", "items")]


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    return f"{v:.1f}"


def main():
    cache, names = Path(sys.argv[1]), sys.argv[2:]
    for n in names:
        d = json.loads((cache / f"{n}.json").read_text(encoding="utf-8"))
        s, t = d["state"], d["times"]
        print(f"== {n}  windows={d['windows']} errors={d['errors']} perception={d.get('perception_sec')}s")
        print("    t " + " ".join(f"{short:>6s}" for _, short in KEYS))
        for i in range(len(t)):
            print(f"{t[i]:5.1f} " + " ".join(f"{fmt(s[k][i]):>6s}" for k, _ in KEYS))
        print("events:", [(round(e["t"], 1), e["event"], e["actor"]) for e in d["events"]])
        print("other:", d.get("other"))


if __name__ == "__main__":
    main()
