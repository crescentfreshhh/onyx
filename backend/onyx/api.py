import shutil
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from . import config, db, media, modelstore, pipeline, previews
from .models import JobCreate, ModelImport, PresetCreate, PreviewCreate, settings_tag
from .queue import worker

router = APIRouter(prefix="/api")

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov", ".m2ts", ".ts", ".mpg", ".mpeg",
                    ".wmv", ".webm", ".vob", ".m4v", ".flv"}


def _safe_input_path(rel: str) -> Path:
    path = (config.INPUT_DIR / rel.lstrip("/")).resolve()
    if not path.is_relative_to(config.INPUT_DIR.resolve()):
        raise HTTPException(400, "path escapes input directory")
    return path


@router.get("/jobs")
def list_jobs():
    return db.list_jobs()


def _unique_output_path(stem: str, extension: str) -> Path:
    # Never overwrite: skip names taken on disk or reserved by pending jobs.
    reserved = set(db.active_output_paths())
    candidate = config.OUTPUT_DIR / f"{stem}.{extension}"
    n = 1
    while candidate.exists() or str(candidate) in reserved:
        candidate = config.OUTPUT_DIR / f"{stem} ({n}).{extension}"
        n += 1
    return candidate


@router.post("/jobs", status_code=201)
def create_job(body: JobCreate):
    input_path = _safe_input_path(body.input_path)
    if not input_path.is_file():
        raise HTTPException(404, f"input file not found: {body.input_path}")

    container = body.settings.encode.container
    name = body.output_name or f"{input_path.stem}_onyx"
    stem = Path(name).stem
    if body.settings.encode.tag_filename:
        stem = f"{stem}_{settings_tag(body.settings)}"
    output_path = _unique_output_path(stem, container)

    job = db.create_job(str(input_path), str(output_path), body.settings.model_dump())
    worker.notify()
    return job


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: int):
    if not worker.cancel_job(job_id):
        raise HTTPException(409, "job is not queued or running")
    return {"ok": True}


@router.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: int):
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if job["status"] == "running":
        raise HTTPException(409, "cancel the job before deleting it")
    db.delete_job(job_id)


@router.get("/files")
def list_files(path: str = ""):
    base = _safe_input_path(path)
    if not base.is_dir():
        raise HTTPException(404, "no such directory")
    entries = []
    for child in sorted(base.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if child.name.startswith("."):
            continue
        if child.is_dir():
            entries.append({"name": child.name, "type": "dir"})
        elif child.suffix.lower() in VIDEO_EXTENSIONS:
            entries.append({"name": child.name, "type": "file",
                            "size": child.stat().st_size})
    return {"path": path, "entries": entries}


@router.get("/files/stream")
def stream_file(path: str):
    file = _safe_input_path(path)
    if not file.is_file():
        raise HTTPException(404, "no such file")
    return FileResponse(file)


@router.get("/media/info")
async def media_info(path: str):
    file = _safe_input_path(path)
    if not file.is_file():
        raise HTTPException(404, "no such file")
    info = await media.probe(str(file))
    if info is None:
        raise HTTPException(422, "ffprobe could not read this file")
    return info


@router.post("/preview", status_code=202)
async def create_preview(body: PreviewCreate):
    input_path = _safe_input_path(body.input_path)
    if not input_path.is_file():
        raise HTTPException(404, f"input file not found: {body.input_path}")
    preview_id = previews.start(str(input_path), body.settings, body.start_seconds, body.duration)
    return {"id": preview_id}


@router.get("/preview/{preview_id}")
def preview_status(preview_id: str):
    preview = previews.get(preview_id)
    if preview is None:
        raise HTTPException(404, "no such preview")
    return preview


@router.delete("/preview/{preview_id}", status_code=204)
def delete_preview(preview_id: str):
    previews.delete(preview_id)


@router.get("/preview/{preview_id}/{side}")
def preview_clip(preview_id: str, side: str):
    if side not in ("original", "processed"):
        raise HTTPException(404, "side must be 'original' or 'processed'")
    preview = previews.get(preview_id)
    if preview is None or preview["status"] != "ready":
        raise HTTPException(404, "preview not ready")
    file = previews.clip_path(preview_id, side)
    if not file.is_file():
        raise HTTPException(404, "clip missing")
    return FileResponse(file, media_type="video/mp4")


@router.get("/presets")
def list_presets():
    return db.list_presets()


@router.post("/presets", status_code=201)
def save_preset(body: PresetCreate):
    db.save_preset(body.name, body.settings.model_dump())
    return {"ok": True}


@router.delete("/presets/{preset_id}", status_code=204)
def delete_preset(preset_id: int):
    db.delete_preset(preset_id)


@router.get("/models")
def list_models():
    return pipeline.stage_models()


@router.get("/models/catalog")
def models_catalog():
    return modelstore.catalog()


@router.post("/models/{model_id}/download", status_code=202)
def download_model(model_id: str):
    try:
        modelstore.start_download(model_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return {"ok": True}


@router.post("/models/import", status_code=202)
def import_model(body: ModelImport):
    try:
        model_id = modelstore.start_import(body.url)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"id": model_id}


@router.post("/models/{model_id}/convert", status_code=202)
def convert_model(model_id: str):
    try:
        modelstore.start_convert(model_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return {"ok": True}


@router.get("/system")
def system_info():
    gpu = None
    gpu_error = None
    if shutil.which("nvidia-smi") is None:
        gpu_error = ("nvidia-smi not present in container — the NVIDIA runtime is not "
                     "applied. Check '--runtime=nvidia' in Extra Parameters and the "
                     "Nvidia Driver plugin (driver 570+ required for this image).")
    else:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                gpu = out.stdout.strip().splitlines()[0]
            else:
                detail = (out.stderr or out.stdout or "").strip()[-300:]
                gpu_error = f"nvidia-smi failed (driver/runtime mismatch?): {detail}"
        except OSError as exc:
            gpu_error = f"nvidia-smi could not run: {exc}"
    return {
        "version": config.VERSION,
        "gpu": gpu,
        "gpu_error": gpu_error,
        "ffmpeg": shutil.which(config.FFMPEG) is not None,
    }
