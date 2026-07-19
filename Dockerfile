# CUDA 12.8 runtime covers Ampere (sm_86) through Blackwell (sm_120) for the
# upcoming AI inference backends; ffmpeg comes from Ubuntu for now.
# Declared before the first stage so it is usable in FROM lines.
ARG BASE_IMAGE=nvidia/cuda:12.8.0-runtime-ubuntu24.04

FROM node:22-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-venv ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt ./
# torch+torchvision come from the CPU wheel index (conversion-only workload —
# ONNX Runtime handles GPU inference); installing both together prevents pip
# from swapping in the multi-GB CUDA torch build as torchvision's dependency.
RUN python3 -m venv /venv \
    && /venv/bin/pip install --no-cache-dir -r requirements.txt onnxruntime-gpu \
    && /venv/bin/pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && /venv/bin/pip install --no-cache-dir spandrel onnx

COPY backend/onyx ./onyx
COPY --from=ui /ui/dist ./static
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV ONYX_STATIC_DIR=/app/static \
    ONYX_PORT=8484 \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility

EXPOSE 8484
VOLUME ["/config", "/input", "/output"]

ENTRYPOINT ["/entrypoint.sh"]
