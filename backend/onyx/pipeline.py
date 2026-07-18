"""Builds and runs the enhancement pipeline for a job.

v0 engine: every stage maps to an FFmpeg filter so the queue works end to end
on any hardware. AI stages (ONNX/torch frame servers for ESRGAN, RIFE,
SeedVR2) plug in behind the same stage names in a later milestone — the
settings schema and job flow do not change.
"""

import asyncio
import time
from typing import Awaitable, Callable, Optional

from . import config
from .models import JobSettings

# Stage catalog surfaced to the UI. engine="ffmpeg" entries are the v0
# placeholders; AI engines will register alongside them.
STAGE_MODELS = {
    "enhance": [
        {"id": "lanczos", "name": "Lanczos (fast, non-AI)", "engine": "ffmpeg"},
    ],
    "interpolate": [
        {"id": "dup", "name": "Frame duplication (non-AI)", "engine": "ffmpeg"},
        {"id": "minterpolate", "name": "Motion interpolation (slow, non-AI)", "engine": "ffmpeg"},
    ],
    "deinterlace": [
        {"id": "bwdif", "name": "BWDIF", "engine": "ffmpeg"},
        {"id": "yadif", "name": "Yadif", "engine": "ffmpeg"},
    ],
}

ENCODERS = {
    "libx264": ["-c:v", "libx264", "-preset", "slow", "-crf"],
    "libx265": ["-c:v", "libx265", "-preset", "medium", "-crf"],
    "h264_nvenc": ["-c:v", "h264_nvenc", "-preset", "p5", "-cq"],
    "hevc_nvenc": ["-c:v", "hevc_nvenc", "-preset", "p5", "-cq"],
}


def build_filters(settings: JobSettings) -> list[str]:
    filters: list[str] = []
    if settings.deinterlace.enabled:
        filters.append(settings.deinterlace.engine)
    if settings.enhance.enabled and settings.enhance.scale > 1:
        s = settings.enhance.scale
        filters.append(f"scale=iw*{s}:ih*{s}:flags=lanczos")
    if settings.interpolate.enabled:
        fps = settings.interpolate.fps
        if settings.interpolate.model == "minterpolate":
            filters.append(f"minterpolate=fps={fps}:mi_mode=mci")
        else:
            filters.append(f"fps={fps}")
    if settings.grain.enabled and settings.grain.amount > 0:
        filters.append(f"noise=alls={settings.grain.amount}:allf=t")
    return filters


def build_command(input_path: str, output_path: str, settings: JobSettings) -> list[str]:
    cmd = [config.FFMPEG, "-y", "-hide_banner", "-nostats", "-progress", "pipe:1", "-i", input_path]

    filters = build_filters(settings)
    if filters:
        cmd += ["-vf", ",".join(filters)]

    enc = ENCODERS.get(settings.encode.codec, ENCODERS["libx264"])
    cmd += [*enc, str(settings.encode.quality)]
    cmd += ["-pix_fmt", "yuv420p"]

    cmd += ["-map", "0:v:0", "-map", "0:a?"]
    if settings.encode.container == "mkv":
        cmd += ["-map", "0:s?", "-map_chapters", "0", "-c:s", "copy"]
    cmd += ["-c:a", "copy"] if settings.encode.audio == "copy" else ["-c:a", "aac", "-b:a", "192k"]

    cmd.append(output_path)
    return cmd


def output_duration(settings: JobSettings, source_duration: float) -> float:
    return source_duration


async def run(
    input_path: str,
    output_path: str,
    settings: JobSettings,
    source_duration: float,
    on_progress: Callable[[float, Optional[float], Optional[float]], Awaitable[None]],
    cancel_event: asyncio.Event,
) -> None:
    cmd = build_command(input_path, output_path, settings)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    total = output_duration(settings, source_duration)
    started = time.monotonic()
    stderr_task = asyncio.create_task(proc.stderr.read())

    async def watch_cancel() -> None:
        await cancel_event.wait()
        proc.terminate()

    cancel_task = asyncio.create_task(watch_cancel())
    try:
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").strip()
            key, _, value = line.partition("=")
            if key == "out_time_us" and value.lstrip("-").isdigit() and total > 0:
                done = int(value) / 1_000_000
                progress = min(done / total, 1.0)
                elapsed = time.monotonic() - started
                eta = (elapsed / progress - elapsed) if progress > 0.01 else None
                await on_progress(progress, None, eta)
            elif key == "fps" and value:
                try:
                    await on_progress(-1, float(value), None)
                except ValueError:
                    pass
        await proc.wait()
    finally:
        cancel_task.cancel()

    if cancel_event.is_set():
        raise asyncio.CancelledError()
    if proc.returncode != 0:
        stderr = (await stderr_task).decode(errors="replace")
        raise RuntimeError(f"ffmpeg exited with {proc.returncode}: {stderr[-2000:]}")
