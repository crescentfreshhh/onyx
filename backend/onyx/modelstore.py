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
import urllib.parse
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
        "urls": [
            "https://huggingface.co/Phips/2xNomosUni_compact_multijpg/resolve/main/2xNomosUni_compact_multijpg_fp32_opset17.onnx",
            "https://huggingface.co/Phips/2xNomosUni_compact_multijpg/resolve/main/onnx/2xNomosUni_compact_multijpg_fp32_opset17.onnx",
            "https://huggingface.co/Phips/2xNomosUni_compact_multijpg/resolve/main/2xNomosUni_compact_multijpg.onnx",
            "https://huggingface.co/Phips/2xNomosUni_compact_multijpg/resolve/main/onnx/2xNomosUni_compact_multijpg.onnx",
        ],
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
        "urls": [
            "https://huggingface.co/Phips/2xHFA2k_LUDVAE_compact/resolve/main/2xHFA2k_LUDVAE_compact_fp32_opset17.onnx",
            "https://huggingface.co/Phips/2xHFA2k_LUDVAE_compact/resolve/main/onnx/2xHFA2k_LUDVAE_compact_fp32_opset17.onnx",
            "https://huggingface.co/Phips/2xHFA2k_LUDVAE_compact/resolve/main/2xHFA2k_LUDVAE_compact.onnx",
            "https://huggingface.co/Phips/2xHFA2k_LUDVAE_compact/resolve/main/onnx/2xHFA2k_LUDVAE_compact.onnx",
        ],
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

PTH_DESCRIPTION = (
    "PyTorch checkpoint — convert it to ONNX to use it. Conversion runs once "
    "and keeps the original file."
)

_downloads: dict[str, dict[str, Any]] = {}
_imports: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _custom_models() -> list[dict[str, Any]]:
    manifest_files = {m["filename"] for m in MANIFEST}
    models = []
    if not config.MODELS_DIR.is_dir():
        return models
    onnx_stems = []
    for path in sorted(config.MODELS_DIR.glob("*.onnx")):
        onnx_stems.append(path.stem)
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
            "sha256": None,
            "license": "unknown",
            "best_for": "See model source",
            "description": CUSTOM_DESCRIPTION,
        })
    for path in sorted(config.MODELS_DIR.glob("*.pth")):
        stem = path.stem
        # Hide checkpoints that already have a converted ONNX counterpart.
        if any(s == stem or s.endswith(f"x_{stem}") for s in onnx_stems):
            continue
        match = re.match(r"(\d+)x", stem.lower())
        models.append({
            "id": f"pth:{path.name}",
            "name": f"{stem} (PyTorch checkpoint)",
            "stage": "enhance",
            "engine": "onnx",
            "kind": "pth",
            "scale": int(match.group(1)) if match else 2,
            "filename": path.name,
            "sha256": None,
            "license": "unknown",
            "best_for": "See model source",
            "description": PTH_DESCRIPTION,
        })
    return models


def _all_entries() -> list[dict[str, Any]]:
    custom = _custom_models()
    custom_files = {m["filename"] for m in custom}
    with _lock:
        # In-flight URL imports that have not landed on disk yet.
        pending = [e for e in _imports.values() if e["filename"] not in custom_files]
    return MANIFEST + custom + pending


def catalog() -> list[dict[str, Any]]:
    out = []
    for entry in _all_entries():
        item = {k: entry[k] for k in ("id", "name", "stage", "engine", "scale", "license",
                                      "best_for", "description")}
        item["kind"] = entry.get("kind", "onnx")
        path = config.MODELS_DIR / entry["filename"]
        with _lock:
            download = _downloads.get(entry["id"])
        if download and download["status"] in ("downloading", "converting"):
            item["status"] = download["status"]
            item["progress"] = download["progress"]
        elif download and download["status"] == "failed":
            item["status"] = "failed"
            item["error"] = download["error"]
        elif entry.get("kind") == "pth":
            item["status"] = "convertible"
        elif path.is_file():
            item["status"] = "installed"
        elif entry.get("urls"):
            item["status"] = "available"
        else:
            item["status"] = "missing"
        out.append(item)
    return out


def find(model_id: str) -> Optional[dict[str, Any]]:
    for entry in _all_entries():
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
    if entry is None or not entry.get("urls"):
        raise ValueError(f"no downloadable model with id {model_id!r}")
    with _lock:
        active = _downloads.get(model_id)
        if active and active["status"] == "downloading":
            return
        _downloads[model_id] = {"status": "downloading", "progress": 0.0}
    thread = threading.Thread(target=_download, args=(entry,), daemon=True)
    thread.start()


def start_import(url: str) -> str:
    filename = Path(urllib.parse.urlparse(url).path).name
    if not filename.lower().endswith((".onnx", ".pth")):
        raise ValueError("URL must point directly to a .onnx or .pth file")
    filename = filename.replace("/", "_").replace("\\", "_")
    match = re.match(r"(\d+)x", filename.lower())
    entry = {
        "id": f"import:{filename}",
        "name": f"{Path(filename).stem} (imported)",
        "stage": "enhance",
        "engine": "onnx",
        "scale": int(match.group(1)) if match else 2,
        "filename": filename,
        "urls": [url],
        "sha256": None,
        "license": "unknown",
        "best_for": "See model source",
        "description": CUSTOM_DESCRIPTION,
    }
    with _lock:
        active = _downloads.get(entry["id"])
        if active and active["status"] == "downloading":
            return entry["id"]
        _imports[entry["id"]] = entry
        _downloads[entry["id"]] = {"status": "downloading", "progress": 0.0}
    thread = threading.Thread(target=_download, args=(entry,), daemon=True)
    thread.start()
    return entry["id"]


def start_convert(model_id: str) -> None:
    entry = find(model_id)
    if entry is None or entry.get("kind") != "pth":
        raise ValueError(f"no convertible checkpoint with id {model_id!r}")
    with _lock:
        active = _downloads.get(model_id)
        if active and active["status"] == "converting":
            return
        _downloads[model_id] = {"status": "converting", "progress": 0.0}
    thread = threading.Thread(target=_run_convert, args=(entry,), daemon=True)
    thread.start()


def _run_convert(entry: dict[str, Any]) -> None:
    try:
        _convert_file(config.MODELS_DIR / entry["filename"], entry["id"])
    except Exception as exc:
        with _lock:
            _downloads[entry["id"]] = {"status": "failed", "error": str(exc), "progress": 0.0}


def _convert_file(pth_path: Path, model_id: str) -> None:
    """Convert a PyTorch checkpoint to ONNX next to it (spandrel + torch)."""
    try:
        import torch
        from spandrel import ImageModelDescriptor, ModelLoader
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch/spandrel are not installed in this build — cannot convert .pth files"
        ) from exc

    descriptor = ModelLoader().load_from_file(str(pth_path))
    if not isinstance(descriptor, ImageModelDescriptor):
        raise RuntimeError("unsupported checkpoint (not an image-to-image model)")
    if descriptor.input_channels != 3 or descriptor.output_channels != 3:
        raise RuntimeError("only 3-channel RGB models are supported")

    module = descriptor.model.eval()
    scale = descriptor.scale
    stem = pth_path.stem
    target_stem = stem if re.match(r"\d+x", stem.lower()) else f"{scale}x_{stem}"
    target = config.MODELS_DIR / f"{target_stem}.onnx"
    tmp = target.with_suffix(".partial")

    dummy = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        torch.onnx.export(
            module,
            (dummy,),
            str(tmp),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {2: "height", 3: "width"},
                "output": {2: "height_out", 3: "width_out"},
            },
            opset_version=17,
            dynamo=False,
        )
    tmp.rename(target)
    with _lock:
        _downloads[model_id] = {"status": "installed", "progress": 1.0}


def _fetch(url: str, partial: Path, model_id: str) -> str:
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as resp:
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
    return digest.hexdigest()


def _download(entry: dict[str, Any]) -> None:
    model_id = entry["id"]
    target = config.MODELS_DIR / entry["filename"]
    partial = target.with_suffix(".partial")
    last_error: Optional[Exception] = None
    try:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for url in entry["urls"]:
            try:
                digest = _fetch(url, partial, model_id)
                if entry.get("sha256") and digest != entry["sha256"]:
                    raise RuntimeError("sha256 mismatch — refusing to install")
                partial.rename(target)
                if target.suffix == ".pth":
                    with _lock:
                        _downloads[model_id] = {"status": "converting", "progress": 0.0}
                    _convert_file(target, model_id)
                    return
                with _lock:
                    _downloads[model_id] = {"status": "installed", "progress": 1.0}
                return
            except Exception as exc:
                last_error = exc
                partial.unlink(missing_ok=True)
        raise RuntimeError(f"all {len(entry['urls'])} source(s) failed; last error: {last_error}")
    except Exception as exc:
        partial.unlink(missing_ok=True)
        with _lock:
            _downloads[model_id] = {"status": "failed", "error": str(exc), "progress": 0.0}
