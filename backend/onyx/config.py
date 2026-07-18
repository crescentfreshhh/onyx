import os
from pathlib import Path

VERSION = "0.1.0"


def _dir(env_var: str, default: str, fallback: str) -> Path:
    path = Path(os.environ.get(env_var, default))
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError:
        local = Path(fallback)
        local.mkdir(parents=True, exist_ok=True)
        return local


CONFIG_DIR = _dir("ONYX_CONFIG_DIR", "/config", "data/config")
INPUT_DIR = _dir("ONYX_INPUT_DIR", "/input", "data/input")
OUTPUT_DIR = _dir("ONYX_OUTPUT_DIR", "/output", "data/output")

DB_PATH = CONFIG_DIR / "onyx.db"
MODELS_DIR = CONFIG_DIR / "models"

PORT = int(os.environ.get("ONYX_PORT", "8484"))
STATIC_DIR = Path(os.environ.get("ONYX_STATIC_DIR", "static"))
WORKER_ENABLED = os.environ.get("ONYX_DISABLE_WORKER", "") != "1"
FFMPEG = os.environ.get("ONYX_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("ONYX_FFPROBE", "ffprobe")
