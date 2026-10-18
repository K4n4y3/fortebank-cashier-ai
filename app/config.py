import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


ROOT = Path(__file__).resolve().parent.parent
_load_dotenv(ROOT / ".env")

API_KEY = os.environ.get("API_KEY", "")
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data")).resolve()
PIPELINE = os.environ.get("PIPELINE", "full")
WORKERS = int(os.environ.get("WORKERS", "1"))
JOB_DEADLINE_SEC = float(os.environ.get("JOB_DEADLINE_SEC", "270"))
UI_OPEN = os.environ.get("UI_OPEN", "1") == "1"
MAX_UPLOAD_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "200")) * 1024 * 1024)
# Через сколько часов после завершения удалять видео и результат (данные — только в рамках хакатона); 0 — не удалять
RETENTION_HOURS = float(os.environ.get("RETENTION_HOURS", "72"))
WEB_DIR = ROOT / "web"

ALLOWED_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
