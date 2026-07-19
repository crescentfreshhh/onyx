"""ONNX frame-server engines.

Decode FFmpeg → raw RGB frames over a pipe → ONNX inference → raw frames into
an encode FFmpeg that muxes audio/subtitles/chapters back in from the source.
No intermediate frame files are ever written.

Two model kinds run here:
- upscalers: one frame in, one (scaled) frame out
- interpolators (RIFE-class): two frames + a timestep t in (0,1) out to an
  intermediate frame. Output frames are sampled at exact fractional source
  positions, so any source→target fps ratio works directly.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

import numpy as np

from . import config
from .models import JobSettings
from .pipeline import ENCODERS, PREVIEW_ENCODE, post_filters, pre_filters

# TensorRT is deliberately absent: requesting it without the TensorRT
# libraries raises at session creation, whereas CUDA degrades to CPU with a
# logged warning. Revisit when the image actually ships TensorRT.
PROVIDER_PREFERENCE = [
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]

# Interpolating across a hard cut produces ghosting; above this mean-abs-diff
# (0-1 scale) the nearest source frame is duplicated instead.
SCENE_CHANGE_THRESHOLD = 0.12

log = logging.getLogger("onyx.engines")


def _make_session(model_path: Path):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            f"onnxruntime failed to load ({exc}) — AI models are unavailable in this build"
        ) from exc
    available = ort.get_available_providers()
    providers = [p for p in PROVIDER_PREFERENCE if p in available] or available
    # Greedy arena growth causes spurious allocation failures with large
    # (4K-class) tensors; allocate only what each request needs.
    sess_options = ort.SessionOptions()
    sess_options.enable_cpu_mem_arena = False
    provider_options = [
        {"arena_extend_strategy": "kSameAsRequested"} if p == "CUDAExecutionProvider" else {}
        for p in providers
    ]
    session = ort.InferenceSession(
        str(model_path), sess_options=sess_options,
        providers=providers, provider_options=provider_options,
    )
    log.info("model %s active providers: %s", model_path.name, session.get_providers())
    return session


def _to_nchw(frame: np.ndarray) -> np.ndarray:
    x = frame.astype(np.float32) / 255.0
    return np.transpose(x, (2, 0, 1))[np.newaxis]


def _to_uint8(nchw: np.ndarray) -> np.ndarray:
    y = np.clip(nchw[0], 0.0, 1.0)
    return (np.transpose(y, (1, 2, 0)) * 255.0).round().astype(np.uint8)


class OnnxUpscaler:
    def __init__(self, model_path: Path):
        self.session = _make_session(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def upscale(self, frame: np.ndarray) -> np.ndarray:
        y = self.session.run(None, {self.input_name: _to_nchw(frame)})[0]
        return _to_uint8(y)

    def probe_scale(self, width: int, height: int) -> int:
        out = self.upscale(np.zeros((height, width, 3), dtype=np.uint8))
        scale = out.shape[1] // width
        if scale < 1 or out.shape[0] != height * scale or out.shape[1] != width * scale:
            raise RuntimeError(f"model produced unexpected output shape {out.shape}")
        return scale


class OnnxInterpolator:
    """RIFE-class arbitrary-timestep interpolator.

    Supported input layouts (introspected from the graph):
    - three inputs, positional: img0 [1,3,H,W], img1 [1,3,H,W], timestep
      (scalar [1] or a [1,1,H,W] map)
    - one input [1,7,H,W]: img0 RGB + img1 RGB + timestep plane (vs-mlrt v1)
    - one input [1,11,H,W]: the vs-mlrt v2 representation — img0, img1,
      timestep, horizontal/vertical warp grids (2x/(W-1)-1 style) and the
      two flow-normalization constant planes (2/(W-1), 2/(H-1)). v2 models
      handle spatial padding internally, so none is applied here.

    For the other layouts, spatial dims are reflect-padded to a multiple of
    64 and cropped back, matching RIFE's alignment requirement.
    """

    PAD = 64

    def __init__(self, model_path: Path):
        self.session = _make_session(model_path)
        inputs = self.session.get_inputs()
        if len(inputs) >= 3:
            self.mode = "triple"
            self.names = [i.name for i in inputs[:3]]
            self.t_rank = len(inputs[2].shape)
        elif len(inputs) == 1 and inputs[0].shape[1] == 7:
            self.mode = "concat7"
            self.names = [inputs[0].name]
        elif len(inputs) == 1 and inputs[0].shape[1] == 11:
            self.mode = "concat11"
            self.names = [inputs[0].name]
        else:
            shapes = [(i.name, i.shape) for i in inputs]
            raise RuntimeError(f"unsupported interpolator input layout: {shapes}")

    @staticmethod
    def build_v2_input(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        h, w = a.shape[2], a.shape[3]
        t_plane = np.full((1, 1, h, w), t, dtype=np.float32)
        grid_h = np.broadcast_to(
            np.linspace(-1.0, 1.0, w, dtype=np.float32), (1, 1, h, w)
        )
        grid_v = np.broadcast_to(
            np.linspace(-1.0, 1.0, h, dtype=np.float32).reshape(1, 1, h, 1), (1, 1, h, w)
        )
        mult_h = np.full((1, 1, h, w), 2.0 / (w - 1), dtype=np.float32)
        mult_w = np.full((1, 1, h, w), 2.0 / (h - 1), dtype=np.float32)
        return np.concatenate([a, b, t_plane, grid_h, grid_v, mult_h, mult_w], axis=1)

    def _pad(self, x: np.ndarray) -> tuple[np.ndarray, int, int]:
        h, w = x.shape[2], x.shape[3]
        ph = (self.PAD - h % self.PAD) % self.PAD
        pw = (self.PAD - w % self.PAD) % self.PAD
        if ph or pw:
            x = np.pad(x, ((0, 0), (0, 0), (0, ph), (0, pw)), mode="reflect")
        return x, h, w

    def interpolate(self, frame_a: np.ndarray, frame_b: np.ndarray, t: float) -> np.ndarray:
        if self.mode == "concat11":
            a = _to_nchw(frame_a)
            b = _to_nchw(frame_b)
            h, w = a.shape[2], a.shape[3]
            feeds = {self.names[0]: self.build_v2_input(a, b, t)}
        else:
            a, h, w = self._pad(_to_nchw(frame_a))
            b, _, _ = self._pad(_to_nchw(frame_b))
            if self.mode == "triple":
                if self.t_rank >= 4:
                    t_val = np.full((1, 1, a.shape[2], a.shape[3]), t, dtype=np.float32)
                else:
                    t_val = np.array([t], dtype=np.float32)
                feeds = {self.names[0]: a, self.names[1]: b, self.names[2]: t_val}
            else:
                t_plane = np.full((1, 1, a.shape[2], a.shape[3]), t, dtype=np.float32)
                feeds = {self.names[0]: np.concatenate([a, b, t_plane], axis=1)}
        y = self.session.run(None, feeds)[0][:, :, :h, :w]
        return _to_uint8(y)


def scene_change(frame_a: np.ndarray, frame_b: np.ndarray,
                 threshold: float = SCENE_CHANGE_THRESHOLD) -> bool:
    a = frame_a[::8, ::8].astype(np.int16)
    b = frame_b[::8, ::8].astype(np.int16)
    return float(np.abs(a - b).mean()) / 255.0 > threshold


def decode_command(
    input_path: str,
    settings: JobSettings,
    segment: Optional[tuple[float, float]] = None,
) -> list[str]:
    cmd = [config.FFMPEG, "-v", "error"]
    if segment:
        cmd += ["-ss", str(segment[0]), "-t", str(segment[1])]
    cmd += ["-i", input_path]
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
    browser_preview: bool = False,
    ai_interpolated: bool = False,
) -> list[str]:
    cmd = [
        config.FFMPEG, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{out_width}x{out_height}", "-r", str(fps),
        "-i", "pipe:0",
        "-i", input_path,
    ]
    filters = post_filters(settings, skip_interpolate=ai_interpolated)
    if filters:
        cmd += ["-vf", ",".join(filters)]

    if browser_preview:
        cmd += [*PREVIEW_ENCODE, "-map", "0:v:0"]
    else:
        enc = ENCODERS.get(settings.encode.codec, ENCODERS["libx264"])
        cmd += [*enc, str(settings.encode.quality), "-pix_fmt", "yuv420p"]

        cmd += ["-map", "0:v:0", "-map", "1:a?"]
        if settings.encode.container == "mkv":
            cmd += ["-map", "1:s?", "-map_chapters", "1", "-c:s", "copy"]
        cmd += ["-c:a", "copy"] if settings.encode.audio == "copy" else ["-c:a", "aac", "-b:a", "192k"]

    cmd.append(output_path)
    return cmd


async def run_ai(
    input_path: str,
    output_path: str,
    settings: JobSettings,
    info: dict,
    on_progress: Callable[[float, Optional[float], Optional[float]], Awaitable[None]],
    cancel_event: asyncio.Event,
    enhance_model: Optional[Path] = None,
    interp_model: Optional[Path] = None,
    segment: Optional[tuple[float, float]] = None,
    browser_preview: bool = False,
) -> None:
    width, height, src_fps = info["width"], info["height"], info["fps"]
    if not width or not height or not src_fps:
        raise RuntimeError("could not determine source dimensions/framerate")
    if enhance_model is None and interp_model is None:
        raise RuntimeError("run_ai called without any AI model")

    loop = asyncio.get_running_loop()
    upscaler: Optional[OnnxUpscaler] = None
    interpolator: Optional[OnnxInterpolator] = None
    scale = 1
    if enhance_model is not None:
        upscaler = await loop.run_in_executor(None, OnnxUpscaler, enhance_model)
        scale = await loop.run_in_executor(None, upscaler.probe_scale, 64, 64)
    if interp_model is not None:
        interpolator = await loop.run_in_executor(None, OnnxInterpolator, interp_model)

    out_fps = settings.interpolate.fps if interpolator else src_fps
    out_w, out_h = width * scale, height * scale
    duration = segment[1] if segment else info["duration"]
    expected_out = max(int(duration * out_fps), 1)

    decoder = await asyncio.create_subprocess_exec(
        *decode_command(input_path, settings, segment),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=width * height * 3 * 2,
    )
    encoder = await asyncio.create_subprocess_exec(
        *encode_command(input_path, output_path, settings, out_w, out_h, out_fps,
                        browser_preview, ai_interpolated=interpolator is not None),
        stdin=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    frame_bytes = width * height * 3
    frames_in = 0
    frames_out = 0
    started = time.monotonic()
    dec_err = asyncio.create_task(decoder.stderr.read())
    enc_err = asyncio.create_task(encoder.stderr.read())

    async def watch_cancel() -> None:
        # Terminate both processes the moment cancel is requested so blocked
        # pipe reads/writes unblock immediately instead of after the current
        # (possibly slow) inference step.
        await cancel_event.wait()
        decoder.terminate()
        encoder.terminate()

    cancel_task = asyncio.create_task(watch_cancel())

    async def encoder_failed() -> RuntimeError:
        stderr = (await enc_err).decode(errors="replace")
        return RuntimeError(
            f"encode ffmpeg died (exit {encoder.returncode}): {stderr[-2000:]}"
        )

    async def emit(frame: np.ndarray) -> None:
        nonlocal frames_out
        assert encoder.stdin is not None
        if encoder.returncode is not None or encoder.stdin.is_closing():
            raise await encoder_failed()
        try:
            encoder.stdin.write(frame.tobytes())
            await encoder.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, RuntimeError):
            if cancel_event.is_set():
                raise asyncio.CancelledError()
            await encoder.wait()
            raise await encoder_failed()
        frames_out += 1
        if frames_out % 5 == 0:
            progress = min(frames_out / expected_out, 0.999)
            elapsed = time.monotonic() - started
            proc_fps = frames_out / elapsed if elapsed > 0 else None
            eta = (expected_out - frames_out) / proc_fps if proc_fps else None
            await on_progress(progress, round(proc_fps, 1) if proc_fps else None, eta)

    # When both stages run, interpolation happens at source resolution and
    # each emitted frame is upscaled afterwards — peak memory is ~scale²
    # lower than interpolating at upscaled resolution, and RIFE's motion
    # estimation prefers native-res input anyway. Frames are carried as
    # [raw, upscaled|None] pairs so endpoint frames are upscaled at most once.
    async def upscaled(pair: list) -> np.ndarray:
        if upscaler is None:
            return pair[0]
        if pair[1] is None:
            pair[1] = await loop.run_in_executor(None, upscaler.upscale, pair[0])
        return pair[1]

    # Output frame k sits at source position k * src_fps / out_fps; every
    # position inside interval [i-1, i) is synthesized from that frame pair.
    step = src_fps / out_fps
    prev: Optional[list] = None

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
            current = [frame, None]

            if interpolator is None:
                await emit(await upscaled(current))
            elif prev is not None:
                interval_start = frames_in - 1
                is_cut = (
                    scene_change(prev[0], current[0])
                    if settings.interpolate.scene_detect else False
                )
                while frames_out * step < frames_in:
                    t = frames_out * step - interval_start
                    if t < 1e-3:
                        await emit(await upscaled(prev))
                    elif is_cut:
                        await emit(await upscaled(prev if t < 0.5 else current))
                    else:
                        mid = await loop.run_in_executor(
                            None, interpolator.interpolate, prev[0], current[0], float(t)
                        )
                        if upscaler is not None:
                            mid = await loop.run_in_executor(None, upscaler.upscale, mid)
                        await emit(mid)
            prev = current
            frames_in += 1

        if interpolator is not None and prev is not None:
            total_out = int(frames_in * out_fps / src_fps)
            while frames_out < total_out:
                await emit(await upscaled(prev))

        if cancel_event.is_set():
            raise asyncio.CancelledError()
        encoder.stdin.close()
        await asyncio.gather(decoder.wait(), encoder.wait())
    except asyncio.CancelledError:
        decoder.terminate()
        encoder.terminate()
        await asyncio.gather(decoder.wait(), encoder.wait())
        raise
    finally:
        cancel_task.cancel()
        for task in (dec_err, enc_err):
            if not task.done():
                task.cancel()

    if decoder.returncode != 0:
        stderr = (await dec_err).decode(errors="replace")
        raise RuntimeError(f"decode ffmpeg exited with {decoder.returncode}: {stderr[-2000:]}")
    if encoder.returncode != 0:
        stderr = (await enc_err).decode(errors="replace")
        raise RuntimeError(f"encode ffmpeg exited with {encoder.returncode}: {stderr[-2000:]}")
    if frames_out == 0:
        raise RuntimeError("no frames were produced")
