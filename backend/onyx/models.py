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


class GrainSettings(BaseModel):
    enabled: bool = False
    amount: float = Field(4, ge=0, le=100)


class EncodeSettings(BaseModel):
    codec: str = "libx264"
    quality: int = Field(18, ge=0, le=51)
    container: Literal["mkv", "mp4"] = "mkv"
    audio: Literal["copy", "aac"] = "copy"


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


class PreviewCreate(BaseModel):
    input_path: str
    settings: JobSettings = JobSettings()
    start_seconds: float = Field(0, ge=0)
    duration: float = Field(5, gt=0, le=15)
