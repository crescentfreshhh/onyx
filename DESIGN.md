# Onyx — Self-Hosted AI Video Enhancement Server

**Status:** Draft / initial design
**Target platform:** Docker on Unraid (image built and published via GitHub Actions → GHCR)

## 1. What this is

Onyx is a self-hosted, queue-based AI video enhancement server in the spirit of
Topaz Video AI: upscaling, frame interpolation, deinterlacing, denoising, and
restoration of video files — but built as a headless NAS service with a web UI
instead of a desktop application.

### Why not just use Topaz?

Topaz Video AI is a desktop GUI app with per-seat licensing, no Linux server
build, and no automation story. On a NAS, the natural shape for this workload is
what Tdarr proved for transcoding: a long-running server that chews through a
job queue on the box's GPU, controlled from a browser, driven by watch folders
and an API.

**Where Onyx aims to win:**

- Queue-first, headless operation: submit jobs via web UI, watch folder, or REST
  API; get a webhook/notification when done.
- Open model zoo: community super-resolution and interpolation models improve
  constantly and can be swapped freely.
- Correct media handling: lossless passthrough of audio tracks, subtitles,
  chapters, and metadata — an area where Topaz has historically been weak.
- Resumable chunked jobs: an 8-hour render survives a container restart.
- Free and self-hosted.

**Where Onyx does not claim to win (v1):** Topaz's proprietary models (Proteus,
Iris, Rhea) are trained on large private datasets and have strong temporal
consistency. Open frame-by-frame models can shimmer on degraded real-world
footage, and the strongest open temporal models (BasicVSR++ family) are slow.
Positioning: *the best self-hosted workflow around the best open models*, not
"Topaz quality for free."

## 2. Architecture

Single container. Unraid users strongly prefer one-container apps; no Redis or
external database.

```
┌─────────────────────────────────────────────────────┐
│ Container                                           │
│                                                     │
│  Web UI (React, prebuilt static)                    │
│       │                                             │
│  FastAPI backend ──── SQLite (jobs, presets,        │
│       │                       settings, history)    │
│  Job scheduler / worker supervisor                  │
│       │                                             │
│  Pipeline worker(s)                                 │
│    FFmpeg decode → filter stages → FFmpeg encode    │
│                     (ONNX Runtime / VapourSynth)    │
└─────────────────────────────────────────────────────┘
```

### 2.1 Pipeline engine

- **Demux/decode/encode:** FFmpeg. Hardware decode where available.
- **Frame transport:** raw frames streamed over pipes with bounded queues.
  Never extract frame sequences to disk (the Video2X failure mode: a 2-hour
  1080p film is ~2 TB of PNGs).
- **Filter stack:** ordered, composable stages, mirroring Topaz's filter chain:
  1. Deinterlace (QTGMC via VapourSynth — still the gold standard)
  2. Restore / denoise (Real-ESRGAN restore variants, SCUNet)
  3. Upscale (Real-ESRGAN, compact SRVGGNet models for speed)
  4. Frame interpolation (RIFE 4.x)
  5. Grain synthesis (film grain re-application after cleaning)
- **Chunked processing:** video processed in segments (e.g. 30 s) with
  per-chunk checkpoints; jobs resume after crash/restart. Chunks stitched
  losslessly at encode boundaries (segment-accurate cut points).
- **Encode:** NVENC / QSV / software x264/x265/SVT-AV1. CRF/bitrate controls,
  full passthrough of audio, subtitles, chapters, attachments.

### 2.2 Inference runtime

- **Primary:** ONNX Runtime with CUDA and TensorRT execution providers
  (NVIDIA first — the dominant Unraid GPU).
- **Later:** NCNN-Vulkan backend for AMD/Intel GPUs; CPU fallback for testing.
- **VRAM management:** stages share the GPU; models are loaded/unloaded per
  chunk when VRAM is tight, kept resident when not. Tile-based upscaling for
  large resolutions.

### 2.3 Model manager

- Models are **not** baked into the image. Downloaded on first use into
  `/config/models` from a hash-verified manifest (hosted in the GitHub repo,
  releases as CDN).
- Support importing arbitrary community ONNX super-resolution models
  (chaiNNer/OpenModelDB ecosystem compatibility).

### 2.4 Jobs, presets, previews

- **Job:** input file + filter stack config + encode config + output target.
- **Presets:** named filter-stack templates ("VHS restore", "Anime 2x",
  "DVD → 1080p", "24→60fps sports").
- **Preview render (the killer feature):** select any short segment, render it
  through the current stack in seconds/minutes, compare before/after in the
  browser with a wipe slider. Commit to the full render only after previewing.
- **Watch folders:** map a folder to a preset; files dropped in get queued
  automatically.
- **Notifications:** webhooks (Discord/ntfy/generic) on job completion/failure.

### 2.5 API

REST API (OpenAPI-documented, since it's FastAPI for free): submit/cancel jobs,
query progress, manage presets. Enables future Sonarr/Radarr-style integration
and scripting.

## 3. Unraid integration

- **Community Applications template** (XML) in the repo.
- Volume conventions: `/config` (appdata), `/input`, `/output`.
- `PUID` / `PGID` / `UMASK` env vars, linuxserver.io-style.
- GPU: `--runtime=nvidia` + `NVIDIA_VISIBLE_DEVICES` (Nvidia Driver plugin), or
  `/dev/dri` for Intel/AMD later.
- Configurable process/GPU priority so Plex/Jellyfin transcodes are not starved.
- Single web UI port (default 8484), no auth in v0 (LAN assumption), basic auth
  or forward-auth header support later for reverse-proxy users.

## 4. Build & distribution

- **GitHub Actions** → multi-stage Docker build → **GHCR**.
- Tags: `latest` / `vX.Y.Z` (CUDA build), later `-cpu` and `-vulkan` variants.
- Image size discipline: CUDA runtime base is ~3–4 GB; use ONNX Runtime GPU
  rather than full PyTorch in the runtime image, models downloaded at runtime.
- CI: lint + unit tests + a CPU smoke test that runs a tiny clip through the
  full pipeline.

## 5. Tech stack summary

| Layer | Choice | Rationale |
|---|---|---|
| Backend | Python + FastAPI | ML ecosystem, OpenAPI for free |
| Queue/state | SQLite (WAL) | Single container, zero admin |
| Inference | ONNX Runtime (CUDA/TensorRT EP) | Fast, lighter than PyTorch at runtime |
| Deinterlace | VapourSynth + QTGMC | Best-in-class, hard to beat with AI |
| Media I/O | FFmpeg | Everything |
| Web UI | React + Vite, prebuilt static | Served by backend, no Node at runtime |
| Distribution | GitHub Actions → GHCR | As requested |

## 6. Roadmap

- **v0.1 — prove the pipeline:** NVIDIA only. Upscale (Real-ESRGAN + compact
  models) and interpolation (RIFE). Job queue, web UI with job management and
  the preview/comparison slider. NVENC encode with full passthrough.
- **v0.2 — the restoration story:** QTGMC deinterlacing, denoise/restore
  models, grain synthesis, presets, watch folders, webhooks, REST API polish,
  CA template submission.
- **v1.0 — breadth:** AMD/Intel via NCNN-Vulkan, custom ONNX model import,
  multi-GPU scheduling, auth for reverse-proxy setups.
- **Later / research:** temporal-consistency models as they become practical
  (TSCUNet-class), scene-aware auto-preset selection, distributed workers.

## 7. Risks

| Risk | Mitigation |
|---|---|
| CUDA image size | Multi-stage build, ONNX Runtime not PyTorch, models at runtime |
| VRAM contention between stages | Per-chunk model residency, tiling, single-flight GPU scheduler |
| Temporal shimmer vs Topaz | Honest positioning; QTGMC pre-pass; temporal models later |
| Model licensing | Stick to permissive models (Real-ESRGAN BSD, RIFE MIT); manifest records licenses |
| FFmpeg/VapourSynth build complexity | Pin versions, build in CI, cache layers |
