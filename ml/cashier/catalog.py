"""Каталог типов нарушений (violations.yaml): название, пояснение, нормативные источники."""
from functools import lru_cache
from pathlib import Path

import yaml

CATALOG_PATH = Path(__file__).with_name("violations.yaml")


@lru_cache(maxsize=1)
def catalog() -> dict:
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["types"]


def rule(type_id: str) -> dict:
    types = catalog()
    return types.get(type_id, types["OTHER"])


def source_text(type_id: str) -> str:
    sources = rule(type_id).get("sources") or []
    parts = []
    for s in sources:
        item = s["document"]
        if s.get("clause"):
            item += f", {s['clause']}"
        parts.append(item)
    return ("Норматив: " + "; ".join(parts) + ".") if parts else ""
