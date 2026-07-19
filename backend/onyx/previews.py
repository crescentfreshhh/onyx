"""Short before/after preview renders.

A preview renders two browser-playable clips of the same segment: the
untouched source and the processed result of the current filter stack, so the
UI can show a wipe comparison before the user commits to a full render.
"""

import asyncio
import time
import uuid
from typing import Any, Optional

from . import config, engines, media, modelstore, pipeline
from .models import JobSettings

PREVIEW_DIR = config.CONFIG_DIR / "previews"
MAX_AGE_SECONDS = 3600

_previews: dict[str, dict[str, Any]] = {}


def get(preview_id: str) -> Optional[dict[str, Any]]:
    return _previews.get(preview_id)


def clip_path(preview_id: str, side: str):
    return PREVIEW_DIR / f"{preview_id}_{side}.mp4"


def _prune() -> None:
    cutoff = time.time() - MAX_AGE_SECONDS
    for preview_id in [pid for pid, p in _previews.items() if p["created_at"] < cutoff]:
        _previews.pop(preview_id, None)
        for side in ("original", "processed"):
            clip_path(preview_id, side).unlink(missing_ok=True)


async def _noop_progress(progress: float, fps, eta) -> None:
    return None


def start(input_path: str, settings: JobSettings, start_seconds: float, duration: float) -> str:
    _prune()
    preview_id = uuid.uuid4().hex[:12]
    _previews[preview_id] = {
        "id": preview_id,
        "status": "rendering",
        "error": None,
        "created_at": time.time(),
    }
    asyncio.get_running_loop().create_task(
        _render(preview_id, input_path, settings, start_seconds, duration)
    )
    return preview_id


async def _render(
    preview_id: str,
    input_path: str,
    settings: JobSettings,
    start_seconds: float,
    duration: float,
) -> None:
    try:
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        segment = (start_seconds, duration)

        original = clip_path(preview_id, "original")
        cmd = [
            config.FFMPEG, "-y", "-v", "error",
            "-ss", str(start_seconds), "-t", str(duration),
            "-i", input_path,
            *pipeline.PREVIEW_ENCODE,
            str(original),
        ]
        proc = await asyncio.create_subprocess_exec(*cmd, stderr=asyncio.subprocess.PIPE)
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"source clip render failed: {stderr.decode(errors='replace')[-500:]}")

        processed = clip_path(preview_id, "processed")
        from .queue import _resolve_model

        enhance_path = _resolve_model(settings.enhance.model if settings.enhance.enabled else None)
        interp_path = _resolve_model(
            settings.interpolate.model if settings.interpolate.enabled else None
        )
        if enhance_path or interp_path:
            info = await media.probe(input_path)
            if info is None:
                raise RuntimeError("could not probe input file")
            await engines.run_ai(
                input_path, str(processed), settings, info,
                _noop_progress, asyncio.Event(),
                enhance_model=enhance_path, interp_model=interp_path,
                segment=segment, browser_preview=True,
            )
        else:
            await pipeline.run(
                input_path, str(processed), settings, duration,
                _noop_progress, asyncio.Event(),
                segment=segment, browser_preview=True,
            )
        _previews[preview_id]["status"] = "ready"
    except Exception as exc:
        _previews[preview_id]["status"] = "failed"
        _previews[preview_id]["error"] = str(exc)
