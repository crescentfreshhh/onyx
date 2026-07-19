import asyncio
import json
from typing import Any, Optional

from . import config


async def probe(path: str) -> Optional[dict[str, Any]]:
    proc = await asyncio.create_subprocess_exec(
        config.FFPROBE,
        "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        return None
    return parse_probe(json.loads(stdout))


async def validate_output(path: str) -> Optional[str]:
    """Return None if the file is a decodable video, else an error string.

    Catches outputs that exit 0 but are unplayable (e.g. some NVENC/driver
    failures) before a job is reported complete. Decodes the first second
    only — cheap, and enough to expose a bad container/header/stream.
    """
    proc = await asyncio.create_subprocess_exec(
        config.FFMPEG, "-v", "error", "-t", "1", "-i", path, "-f", "null", "-",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        detail = stderr.decode(errors="replace").strip()[-400:]
        return detail or f"ffmpeg could not decode the output (exit {proc.returncode})"
    info = await probe(path)
    if info is None or info["width"] == 0 or info["duration"] <= 0:
        return "output has no valid video stream or zero duration"
    return None


def parse_probe(raw: dict) -> dict[str, Any]:
    video = next((s for s in raw.get("streams", []) if s.get("codec_type") == "video"), {})
    fmt = raw.get("format", {})

    fps = 0.0
    rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
    num, _, den = rate.partition("/")
    if den and float(den) != 0:
        fps = float(num) / float(den)

    return {
        "width": video.get("width", 0),
        "height": video.get("height", 0),
        "fps": round(fps, 3),
        "duration": float(fmt.get("duration", 0) or 0),
        "codec": video.get("codec_name", "unknown"),
        "interlaced": video.get("field_order", "progressive") not in ("progressive", "unknown", None),
        "size_bytes": int(fmt.get("size", 0) or 0),
    }
