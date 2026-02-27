from pydantic import BaseModel
from pydantic import Field


class PersonMetadata(BaseModel):
    id: int
    yaw: float
    pitch: float
    distance: float


class RobotMetadata(BaseModel):
    head_yaw: float = Field(..., description="Current HeadYaw in radians")
    head_pitch: float = Field(..., description="Current HeadPitch in radians")
    body_yaw: float | None = Field(None, description="Body yaw in radians")
    camera_hfov: float | None = Field(None, description="Camera HFOV in degrees")
    camera_vfov: float | None = Field(None, description="Camera VFOV in degrees")
    image_width: int | None = Field(None, description="Image width in pixels")
    image_height: int | None = Field(None, description="Image height in pixels")
    timestamp: float | None = Field(None, description="Capture timestamp (s)")
    frame_id: str | None = Field(None, description="Frame identifier")
    scan_id: str | None = Field(None, description="Scan session identifier")
    people: list[PersonMetadata] = Field(default_factory=list)
    battery: int | None = None  # this might be redundant
