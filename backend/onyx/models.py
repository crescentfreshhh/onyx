from typing import Literal, Optional

from pydantic import BaseModel, Field


class DeinterlaceSettings(BaseModel):
    enabled: bool = False
    engine: str = "bwdif"


class EnhanceSettings(BaseModel):
    enabled: bool = False
    model: str = "lanczos"
    scale: Literal[1, 2, 4] = 2


class InterpolateSettings(BaseModel):
    enabled: bool = False
    model: str = "dup"
    fps: float = Field(60, gt=0, le=480)
    scene_detect: bool = True


class GrainSettings(BaseModel):
    enabled: bool = False
    amount: float = Field(4, ge=0, le=100)


class EncodeSettings(BaseModel):
    codec: str = "libx264"
    quality: int = Field(18, ge=0, le=51)
    container: Literal["mkv", "mp4"] = "mkv"
    audio: Literal["copy", "aac"] = "copy"
    tag_filename: bool = False


def settings_tag(settings: "JobSettings") -> str:
    """Compact settings summary for output filenames, e.g.
    '2x-NomosUni_60fps-rife_v4.6_crf18-libx264'."""
    parts: list[str] = []
    if settings.deinterlace.enabled:
        parts.append(settings.deinterlace.engine)
    if settings.enhance.enabled:
        model = settings.enhance.model.split(":")[-1].rsplit(".", 1)[0]
        parts.append(f"{settings.enhance.scale}x-{model}")
    if settings.interpolate.enabled:
        model = settings.interpolate.model.split(":")[-1].rsplit(".", 1)[0]
        fps = f"{settings.interpolate.fps:g}"
        parts.append(f"{fps}fps-{model}")
    if settings.grain.enabled:
        parts.append(f"grain{settings.grain.amount:g}")
    parts.append(f"crf{settings.encode.quality}-{settings.encode.codec}")
    tag = "_".join(parts)
    return "".join(c if c.isalnum() or c in "._-" else "-" for c in tag)


class JobSettings(BaseModel):
    deinterlace: DeinterlaceSettings = DeinterlaceSettings()
    enhance: EnhanceSettings = EnhanceSettings()
    interpolate: InterpolateSettings = InterpolateSettings()
    grain: GrainSettings = GrainSettings()
    encode: EncodeSettings = EncodeSettings()


class JobCreate(BaseModel):
    input_path: str
    output_name: Optional[str] = None
    settings: JobSettings = JobSettings()


class PresetCreate(BaseModel):
    name: str
    settings: JobSettings


class ModelImport(BaseModel):
    url: str


class PreviewCreate(BaseModel):
    input_path: str
    settings: JobSettings = JobSettings()
    start_seconds: float = Field(0, ge=0)
    duration: float = Field(5, gt=0, le=15)
