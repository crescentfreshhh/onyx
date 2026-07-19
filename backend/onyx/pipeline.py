"""Builds and runs the enhancement pipeline for a job.

v0 engine: every stage maps to an FFmpeg filter so the queue works end to end
on any hardware. AI stages (ONNX/torch frame servers for ESRGAN, RIFE,
SeedVR2) plug in behind the same stage names in a later milestone — the
settings schema and job flow do not change.
"""

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional

from . import config
from .models import JobSettings

log = logging.getLogger("onyx.pipeline")

# Built-in FFmpeg stage engines; installed ONNX models are appended by
# stage_models() at request time.
STAGE_MODELS = {
    "enhance": [
        {
            "id": "lanczos",
            "name": "Lanczos (fast, non-AI)",
            "engine": "ffmpeg",
            "description": "Plain resampler. A fine choice for pristine sources, "
                           "where AI adds little — and it's instant.",
        },
    ],
    "interpolate": [
        {
            "id": "dup",
            "name": "Frame duplication (non-AI)",
            "engine": "ffmpeg",
            "description": "Reaches the target FPS by repeating frames — no motion "
                           "smoothing, but never introduces artifacts.",
        },
        {
            "id": "minterpolate",
            "name": "Motion interpolation (slow, non-AI)",
            "engine": "ffmpeg",
            "description": "Motion-compensated blending. Slow, and can warp on "
                           "complex motion; the RIFE AI engine will supersede it.",
        },
    ],
    "deinterlace": [
        {
            "id": "bwdif",
            "name": "BWDIF",
            "engine": "ffmpeg",
            "description": "Recommended for DVD and broadcast sources.",
        },
        {
            "id": "yadif",
            "name": "Yadif",
            "engine": "ffmpeg",
            "description": "Legacy alternative — prefer BWDIF unless it misbehaves.",
        },
    ],
}


def stage_models() -> dict:
    from . import modelstore

    merged = {stage: list(entries) for stage, entries in STAGE_MODELS.items()}
    for model in modelstore.catalog():
        if model["status"] == "installed":
            merged.setdefault(model["stage"], []).append({
                "id": model["id"],
                "name": model["name"],
                "engine": model["engine"],
                "description": model.get("description"),
                "best_for": model.get("best_for"),
            })
    return merged

# (args before the quality value, args after). NVENC constant-quality needs
# an explicit rate-control mode AND -b:v 0, or it can emit an unplayable
# stream that still exits 0.
ENCODERS: dict[str, tuple[list[str], list[str]]] = {
    "libx264": (["-c:v", "libx264", "-preset", "slow", "-crf"], []),
    "libx265": (["-c:v", "libx265", "-preset", "medium", "-crf"], []),
    "h264_nvenc": (["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq"], ["-b:v", "0"]),
    "hevc_nvenc": (["-c:v", "hevc_nvenc", "-preset", "p5", "-rc", "vbr", "-cq"], ["-b:v", "0"]),
}


def encoder_args(codec: str, quality: int) -> list[str]:
    pre, post = ENCODERS.get(codec, ENCODERS["libx264"])
    return [*pre, str(quality), *post]


def pre_filters(settings: JobSettings) -> list[str]:
    filters: list[str] = []
    if settings.deinterlace.enabled:
        filters.append(settings.deinterlace.engine)
    return filters


# NTSC-family decimals are approximations; use exact rationals so long
# renders don't drift against the audio track.
NTSC_RATES = {
    23.976: "24000/1001",
    29.97: "30000/1001",
    59.94: "60000/1001",
    119.88: "120000/1001",
}


def fps_expr(fps: float) -> str:
    for decimal, rational in NTSC_RATES.items():
        if abs(fps - decimal) < 0.001:
            return rational
    return str(fps)


def post_filters(settings: JobSettings, skip_interpolate: bool = False) -> list[str]:
    filters: list[str] = []
    if settings.interpolate.enabled and not skip_interpolate:
        fps = fps_expr(settings.interpolate.fps)
        if settings.interpolate.model == "minterpolate":
            filters.append(f"minterpolate=fps={fps}:mi_mode=mci")
        else:
            filters.append(f"fps={fps}")
    if settings.grain.enabled and settings.grain.amount > 0:
        filters.append(f"noise=alls={settings.grain.amount}:allf=t")
    return filters


def build_filters(settings: JobSettings) -> list[str]:
    filters = pre_filters(settings)
    if settings.enhance.enabled and settings.enhance.scale > 1:
        s = settings.enhance.scale
        filters.append(f"scale=iw*{s}:ih*{s}:flags=lanczos")
    return filters + post_filters(settings)


# Encode arguments for browser-playable preview clips, regardless of the
# job's own output settings.
PREVIEW_ENCODE = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                  "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an"]


def build_command(
    input_path: str,
    output_path: str,
    settings: JobSettings,
    segment: Optional[tuple[float, float]] = None,
    browser_preview: bool = False,
) -> list[str]:
    cmd = [config.FFMPEG, "-y", "-hide_banner", "-nostats", "-progress", "pipe:1"]
    if segment:
        cmd += ["-ss", str(segment[0]), "-t", str(segment[1])]
    cmd += ["-i", input_path]

    filters = build_filters(settings)
    if filters:
        cmd += ["-vf", ",".join(filters)]

    if browser_preview:
        cmd += PREVIEW_ENCODE
    else:
        cmd += encoder_args(settings.encode.codec, settings.encode.quality)
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
    segment: Optional[tuple[float, float]] = None,
    browser_preview: bool = False,
) -> None:
    cmd = build_command(input_path, output_path, settings, segment, browser_preview)
    log.info("ffmpeg (pipeline): %s", " ".join(cmd))
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
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        transport = getattr(proc.stdout, "_transport", None)
        if transport is not None:
            try:
                transport.close()
            except Exception:
                pass

    cancel_task = asyncio.create_task(watch_cancel())
    try:
        assert proc.stdout is not None
        # ffmpeg emits fps= just before out_time_us= in each progress block;
        # report them in a single update so downstream throttling can never
        # starve progress/ETA in favour of fps.
        last_fps: Optional[float] = None
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").strip()
            key, _, value = line.partition("=")
            if key == "fps" and value:
                try:
                    last_fps = float(value)
                except ValueError:
                    pass
            elif key == "out_time_us" and value.lstrip("-").isdigit() and total > 0:
                done = int(value) / 1_000_000
                progress = min(done / total, 1.0)
                elapsed = time.monotonic() - started
                eta = (elapsed / progress - elapsed) if progress > 0.01 else None
                await on_progress(progress, last_fps, eta)
        await proc.wait()
    finally:
        cancel_task.cancel()

    if cancel_event.is_set():
        raise asyncio.CancelledError()
    if proc.returncode != 0:
        stderr = (await stderr_task).decode(errors="replace")
        raise RuntimeError(f"ffmpeg exited with {proc.returncode}: {stderr[-2000:]}")
