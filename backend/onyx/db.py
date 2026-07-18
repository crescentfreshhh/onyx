import json
import sqlite3
import time
from typing import Any, Optional

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  input_path TEXT NOT NULL,
  output_path TEXT NOT NULL,
  settings TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'queued',
  progress REAL NOT NULL DEFAULT 0,
  fps REAL,
  eta_seconds REAL,
  error TEXT,
  created_at REAL NOT NULL,
  started_at REAL,
  finished_at REAL
);
CREATE TABLE IF NOT EXISTS presets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  settings TEXT NOT NULL,
  builtin INTEGER NOT NULL DEFAULT 0
);
"""

BUILTIN_PRESETS = {
    "DVD → 2x Restore": {
        "deinterlace": {"enabled": True},
        "enhance": {"enabled": True, "model": "lanczos", "scale": 2},
        "encode": {"codec": "libx264", "quality": 17},
    },
    "1080p → 4K": {
        "enhance": {"enabled": True, "model": "lanczos", "scale": 2},
        "encode": {"codec": "libx265", "quality": 20},
    },
    "Smooth 60fps": {
        "interpolate": {"enabled": True, "fps": 60},
        "encode": {"codec": "libx264", "quality": 18},
    },
}


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
        # A crash mid-render leaves jobs stuck in 'running'; requeue on boot.
        conn.execute("UPDATE jobs SET status='queued', progress=0 WHERE status='running'")
        for name, settings in BUILTIN_PRESETS.items():
            conn.execute(
                "INSERT OR IGNORE INTO presets (name, settings, builtin) VALUES (?, ?, 1)",
                (name, json.dumps(settings)),
            )


def _row_to_job(row: sqlite3.Row) -> dict[str, Any]:
    job = dict(row)
    job["settings"] = json.loads(job["settings"])
    return job


def create_job(input_path: str, output_path: str, settings: dict) -> dict:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO jobs (input_path, output_path, settings, created_at) VALUES (?, ?, ?, ?)",
            (input_path, output_path, json.dumps(settings), time.time()),
        )
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (cur.lastrowid,)).fetchone()
        return _row_to_job(row)


def get_job(job_id: int) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None


def list_jobs() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM jobs ORDER BY id DESC").fetchall()
        return [_row_to_job(r) for r in rows]


def next_queued_job() -> Optional[dict]:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY id LIMIT 1"
        ).fetchone()
        return _row_to_job(row) if row else None


def update_job(job_id: int, **fields: Any) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    with connect() as conn:
        conn.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))


def delete_job(job_id: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))


def list_presets() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM presets ORDER BY builtin DESC, name").fetchall()
        out = []
        for r in rows:
            p = dict(r)
            p["settings"] = json.loads(p["settings"])
            p["builtin"] = bool(p["builtin"])
            out.append(p)
        return out


def save_preset(name: str, settings: dict) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO presets (name, settings, builtin) VALUES (?, ?, 0) "
            "ON CONFLICT(name) DO UPDATE SET settings=excluded.settings WHERE builtin=0",
            (name, json.dumps(settings)),
        )


def delete_preset(preset_id: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM presets WHERE id=? AND builtin=0", (preset_id,))
