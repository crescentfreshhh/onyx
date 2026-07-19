"""ONNX frame-server engine.

Decode FFmpeg → raw RGB frames over a pipe → ONNX inference → raw frames into
an encode FFmpeg that muxes audio/subtitles/chapters back in from the source.
No intermediate frame files are ever written.
"""

import asyncio
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

import numpy as np

from . import config
from .models import JobSettings
from .pipeline import ENCODERS, post_filters, pre_filters

PROVIDER_PREFERENCE = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]


class OnnxUpscaler:
    def __init__(self, model_path: Path):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed — AI models are unavailable in this build"
            ) from exc
        available = ort.get_available_providers()
        providers = [p for p in PROVIDER_PREFERENCE if p in available] or available
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def upscale(self, frame: np.ndarray) -> np.ndarray:
        x = frame.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[np.newaxis]
        y = self.session.run(None, {self.input_name: x})[0][0]
        y = np.clip(y, 0.0, 1.0)
        return (np.transpose(y, (1, 2, 0)) * 255.0).round().astype(np.uint8)

    def probe_scale(self, width: int, height: int) -> int:
        out = self.upscale(np.zeros((height, width, 3), dtype=np.uint8))
        scale = out.shape[1] // width
        if scale < 1 or out.shape[0] != height * scale or out.shape[1] != width * scale:
            raise RuntimeError(f"model produced unexpected output shape {out.shape}")
        return scale


def decode_command(input_path: str, settings: JobSettings) -> list[str]:
    cmd = [config.FFMPEG, "-v", "error", "-i", input_path]
    filters = pre_filters(settings)
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    return cmd


def encode_command(
    input_path: str,
    output_path: str,
    settings: JobSettings,
    out_width: int,
    out_height: int,
    fps: float,
) -> list[str]:
    cmd = [
        config.FFMPEG, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{out_width}x{out_height}", "-r", str(fps),
        "-i", "pipe:0",
        "-i", input_path,
    ]
    filters = post_filters(settings)
    if filters:
        cmd += ["-vf", ",".join(filters)]

    enc = ENCODERS.get(settings.encode.codec, ENCODERS["libx264"])
    cmd += [*enc, str(settings.encode.quality), "-pix_fmt", "yuv420p"]

    cmd += ["-map", "0:v:0", "-map", "1:a?"]
    if settings.encode.container == "mkv":
        cmd += ["-map", "1:s?", "-map_chapters", "1", "-c:s", "copy"]
    cmd += ["-c:a", "copy"] if settings.encode.audio == "copy" else ["-c:a", "aac", "-b:a", "192k"]

    cmd.append(output_path)
    return cmd


async def run_onnx(
    input_path: str,
    output_path: str,
    settings: JobSettings,
    info: dict,
    model_path: Path,
    on_progress: Callable[[float, Optional[float], Optional[float]], Awaitable[None]],
    cancel_event: asyncio.Event,
) -> None:
    width, height, fps = info["width"], info["height"], info["fps"]
    if not width or not height or not fps:
        raise RuntimeError("could not determine source dimensions/framerate")

    loop = asyncio.get_running_loop()
    upscaler = await loop.run_in_executor(None, OnnxUpscaler, model_path)
    scale = await loop.run_in_executor(None, upscaler.probe_scale, 64, 64)

    decoder = await asyncio.create_subprocess_exec(
        *decode_command(input_path, settings),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=width * height * 3 * 2,
    )
    encoder = await asyncio.create_subprocess_exec(
        *encode_command(input_path, output_path, settings, width * scale, height * scale, fps),
        stdin=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    frame_bytes = width * height * 3
    expected_frames = max(int(info["duration"] * fps), 1)
    frames = 0
    started = time.monotonic()
    dec_err = asyncio.create_task(decoder.stderr.read())
    enc_err = asyncio.create_task(encoder.stderr.read())

    try:
        assert decoder.stdout is not None and encoder.stdin is not None
        while True:
            if cancel_event.is_set():
                raise asyncio.CancelledError()
            try:
                raw = await decoder.stdout.readexactly(frame_bytes)
            except asyncio.IncompleteReadError:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            out = await loop.run_in_executor(None, upscaler.upscale, frame)
            encoder.stdin.write(out.tobytes())
            await encoder.stdin.drain()

            frames += 1
            if frames % 5 == 0:
                progress = min(frames / expected_frames, 0.999)
                elapsed = time.monotonic() - started
                proc_fps = frames / elapsed if elapsed > 0 else None
                eta = (expected_frames - frames) / proc_fps if proc_fps else None
                await on_progress(progress, round(proc_fps, 1) if proc_fps else None, eta)

        encoder.stdin.close()
        await asyncio.gather(decoder.wait(), encoder.wait())
    except asyncio.CancelledError:
        decoder.terminate()
        encoder.terminate()
        await asyncio.gather(decoder.wait(), encoder.wait())
        raise
    finally:
        for task in (dec_err, enc_err):
            if not task.done():
                task.cancel()

    if decoder.returncode != 0:
        stderr = (await dec_err).decode(errors="replace")
        raise RuntimeError(f"decode ffmpeg exited with {decoder.returncode}: {stderr[-2000:]}")
    if encoder.returncode != 0:
        stderr = (await enc_err).decode(errors="replace")
        raise RuntimeError(f"encode ffmpeg exited with {encoder.returncode}: {stderr[-2000:]}")
    if frames == 0:
        raise RuntimeError("decoder produced no frames")
