from functools import lru_cache

import yaml

from app.config import RULES_PATH


@lru_cache(maxsize=1)
def catalog() -> dict:
    with open(RULES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["types"]


def rule(type_id: str) -> dict:
    types = catalog()
    return types.get(type_id, types["OTHER"])


def source_text(type_id: str) -> str:
    """Нормативная ссылка одной строкой для поля description API."""
    sources = rule(type_id).get("sources") or []
    if not sources:
        return ""
    parts = []
    for s in sources:
        item = s["document"]
        if s.get("clause"):
            item += f", {s['clause']}"
        parts.append(item)
    return "Норматив: " + "; ".join(parts) + "."
