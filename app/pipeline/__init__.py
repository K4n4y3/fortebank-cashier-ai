from importlib import import_module
from types import ModuleType

PIPELINES = {"dummy": "app.pipeline.dummy", "full": "app.pipeline.full"}


def get_pipeline(name: str) -> ModuleType:
    """Модуль конвейера: обязательная analyze(video, progress, deadline), опциональная warmup()."""
    if name not in PIPELINES:
        raise ValueError(f"Unknown PIPELINE={name!r}")
    return import_module(PIPELINES[name])
