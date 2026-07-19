"""Model catalog and downloads.

Two sources of AI models:
- the bundled manifest (downloadable at runtime into MODELS_DIR); URLs are
  data, not code — update MANIFEST as hosts move.
- custom imports: any *.onnx file placed in MODELS_DIR is surfaced as an
  enhance model. A leading "Nx" in the filename sets the scale (chaiNNer /
  OpenModelDB naming convention), defaulting to 2.
"""

import hashlib
import re
import threading
import urllib.request
from pathlib import Path
from typing import Any, Optional

from . import config

# Compact SRVGGNet upscalers, RGB / NCHW / float32 0-1 (chaiNNer convention).
# sha256 is pinned after first verified download; None skips verification.
MANIFEST: list[dict[str, Any]] = [
    {
        "id": "2x-nomosuni-compact",
        "name": "2x NomosUni Compact (universal)",
        "stage": "enhance",
        "engine": "onnx",
        "scale": 2,
        "filename": "2xNomosUni_compact_multijpg.onnx",
        "url": "https://huggingface.co/Phips/2xNomosUni_compact_multijpg/resolve/main/2xNomosUni_compact_multijpg_fp32_opset17.onnx",
        "sha256": None,
        "license": "CC-BY-4.0",
        "best_for": "Live action · compressed sources",
        "description": "The general-purpose default. Trained on compression-degraded "
                       "sources, so it cleans block/ring artifacts while upscaling — best "
                       "for typical mixed-quality live-action video. Not for anime.",
    },
    {
        "id": "2x-hfa2k-compact",
        "name": "2x HFA2k Compact (anime)",
        "stage": "enhance",
        "engine": "onnx",
        "scale": 2,
        "filename": "2xHFA2k_LUDVAE_compact.onnx",
        "url": "https://huggingface.co/Phips/2xHFA2k_LUDVAE_compact/resolve/main/2xHFA2k_LUDVAE_compact_fp32_opset17.onnx",
        "sha256": None,
        "license": "CC-BY-4.0",
        "best_for": "Anime · animation",
        "description": "Animation specialist — excellent on line art and flat color. "
                       "Never use on live action: it gives faces a waxy, plastic look.",
    },
]

CUSTOM_DESCRIPTION = (
    "Imported community model. Check its OpenModelDB page for the content "
    "type it was trained on."
)

_downloads: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _custom_models() -> list[dict[str, Any]]:
    manifest_files = {m["filename"] for m in MANIFEST}
    models = []
    if not config.MODELS_DIR.is_dir():
        return models
    for path in sorted(config.MODELS_DIR.glob("*.onnx")):
        if path.name in manifest_files:
            continue
        match = re.match(r"(\d+)x", path.stem.lower())
        models.append({
            "id": f"custom:{path.name}",
            "name": f"{path.stem} (imported)",
            "stage": "enhance",
            "engine": "onnx",
            "scale": int(match.group(1)) if match else 2,
            "filename": path.name,
            "url": None,
            "sha256": None,
            "license": "unknown",
            "best_for": "See model source",
            "description": CUSTOM_DESCRIPTION,
        })
    return models


def catalog() -> list[dict[str, Any]]:
    out = []
    for entry in MANIFEST + _custom_models():
        item = {k: entry[k] for k in ("id", "name", "stage", "engine", "scale", "license",
                                      "best_for", "description")}
        path = config.MODELS_DIR / entry["filename"]
        with _lock:
            download = _downloads.get(entry["id"])
        if path.is_file():
            item["status"] = "installed"
        elif download and download["status"] == "downloading":
            item["status"] = "downloading"
            item["progress"] = download["progress"]
        elif download and download["status"] == "failed":
            item["status"] = "failed"
            item["error"] = download["error"]
        elif entry.get("url"):
            item["status"] = "available"
        else:
            item["status"] = "missing"
        out.append(item)
    return out


def find(model_id: str) -> Optional[dict[str, Any]]:
    for entry in MANIFEST + _custom_models():
        if entry["id"] == model_id:
            return entry
    return None


def installed_path(model_id: str) -> Optional[Path]:
    entry = find(model_id)
    if entry is None:
        return None
    path = config.MODELS_DIR / entry["filename"]
    return path if path.is_file() else None


def start_download(model_id: str) -> None:
    entry = find(model_id)
    if entry is None or not entry.get("url"):
        raise ValueError(f"no downloadable model with id {model_id!r}")
    with _lock:
        active = _downloads.get(model_id)
        if active and active["status"] == "downloading":
            return
        _downloads[model_id] = {"status": "downloading", "progress": 0.0}
    thread = threading.Thread(target=_download, args=(entry,), daemon=True)
    thread.start()


def _download(entry: dict[str, Any]) -> None:
    model_id = entry["id"]
    target = config.MODELS_DIR / entry["filename"]
    partial = target.with_suffix(".partial")
    try:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with urllib.request.urlopen(entry["url"], timeout=60) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            with open(partial, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    if total:
                        with _lock:
                            _downloads[model_id]["progress"] = round(done / total, 3)
        if entry.get("sha256") and digest.hexdigest() != entry["sha256"]:
            raise RuntimeError("sha256 mismatch — refusing to install")
        partial.rename(target)
        with _lock:
            _downloads[model_id] = {"status": "installed", "progress": 1.0}
    except Exception as exc:
        partial.unlink(missing_ok=True)
        with _lock:
            _downloads[model_id] = {"status": "failed", "error": str(exc), "progress": 0.0}
