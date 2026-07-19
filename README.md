# Onyx

Self-hosted AI video enhancement server for Unraid and Docker — upscaling,
frame interpolation, deinterlacing and restoration with a queue-based workflow
and a familiar filter-stack web UI.

> **Status: early development.** The queue, pipeline and web UI are functional.
> AI upscaling runs through an ONNX frame-server engine (FFmpeg decode → ONNX
> Runtime inference → FFmpeg encode, no intermediate frame files): download
> compact SRVGGNet models from the in-app catalog, or drop any community
> `.onnx` upscaler (OpenModelDB/chaiNNer ecosystem) into `/config/models`.
> PyTorch checkpoints (`.pth` — the common OpenModelDB format) are converted
> to ONNX in-app via spandrel: drop the file in `/config/models` or paste its
> URL, hit Convert, done. Frame interpolation supports RIFE-class ONNX models
> with true arbitrary timesteps (any source→target FPS), scene-change
> detection, and combined upscale+interpolate in a single pass — import a
> RIFE ONNX build (e.g. from vs-mlrt releases) via URL or drop it in
> `/config/models`. Deinterlacing still uses FFmpeg engines; SeedVR2-class
> diffusion restoration (see [MODELS.md](MODELS.md)) is next. Catalog
> download URLs are provisional until first-download verification.

## Features

- Job queue: add videos, configure a filter stack, let the box render overnight
- Filter stack: deinterlace → enhance/upscale → frame interpolation → grain
- Presets (built-in + user-defined)
- Full stream passthrough: audio, subtitles and chapters are preserved (MKV)
- Progress, fps and ETA per job; cancel/requeue; crash-safe (running jobs
  requeue on restart)
- REST API with OpenAPI docs at `/docs`

## Quick start (Docker)

```bash
docker run -d --name onyx \
  -p 8484:8484 \
  -v /path/to/appdata:/config \
  -v /path/to/videos:/input:ro \
  -v /path/to/rendered:/output \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  ghcr.io/crescentfreshhh/onyx:latest
```

Open `http://<host>:8484`.

## Unraid

A Community Applications template lives in [`unraid/onyx.xml`](unraid/onyx.xml).
Until it's in CA: Docker tab → Add Container → paste the template URL.
For GPU support install the **Nvidia Driver** plugin and add
`--runtime=nvidia` to Extra Parameters.

| Path | Purpose |
|---|---|
| `/config` | Database, settings, downloaded models |
| `/input` | Source videos (read-only) |
| `/output` | Rendered videos |

Environment: `PUID` / `PGID` / `UMASK` (linuxserver.io conventions),
`ONYX_PORT` (default 8484).

## Development

```bash
# backend
cd backend
pip install -r requirements.txt pytest httpx
python -m pytest tests -q
ONYX_CONFIG_DIR=./data/config ONYX_INPUT_DIR=./data/input \
ONYX_OUTPUT_DIR=./data/output python -m uvicorn onyx.main:app --port 8484

# frontend (dev server proxies /api to :8484)
cd frontend
npm install
npm run dev
```

## Documentation

- [DESIGN.md](DESIGN.md) — architecture and roadmap
- [MODELS.md](MODELS.md) — open-model survey and hardware targets
