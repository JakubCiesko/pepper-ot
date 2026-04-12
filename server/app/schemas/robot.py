from pydantic import BaseModel
from pydantic import Field


class PersonMetadata(BaseModel):
    id: int
    yaw: float
    pitch: float
    distance: float


class SocialPersonMetadata(BaseModel):
    id: int
    timestamp: float | None = None
    engagement_zone: int | None = None
    is_waving_left: bool | None = None
    is_waving_center: bool | None = None
    is_waving_right: bool | None = None
    is_waving: bool | None = None
    is_sitting: bool | None = None
    is_looking_at_robot: bool | None = None
    looking_at_robot_score: float | None = None
    head_angles: list[float] = Field(default_factory=list)
    gaze_direction: list[str] = Field(default_factory=list)
    gender_code: float | None = None
    gender: str | None = None
    gender_confidence: float | None = None
    age: float | None = None
    age_bucket: str | None = None
    age_confidence: float | None = None
    expression_scores: list[float] = Field(default_factory=list)
    expression: str | None = None
    expression_confidence: float | None = None
    smile_score: float | None = None
    smile_confidence: float | None = None
    eyes_opened: list[float] = Field(default_factory=list)


class RobotMetadata(BaseModel):
    head_yaw: float = Field(..., description="Current HeadYaw in radians")
    head_pitch: float = Field(..., description="Current HeadPitch in radians")
    body_yaw: float | None = Field(None, description="Body yaw in radians")
    camera_hfov: float | None = Field(None, description="Camera HFOV in radians")
    camera_vfov: float | None = Field(None, description="Camera VFOV in radians")
    image_width: int | None = Field(None, description="Image width in pixels")
    image_height: int | None = Field(None, description="Image height in pixels")
    timestamp: float | None = Field(None, description="Capture timestamp (s)")
    frame_id: str | None = Field(None, description="Frame identifier")
    scan_id: str | None = Field(None, description="Scan session identifier")
    capture_mode: str | None = Field(None, description="Capture mode, e.g. scan/single")
    people: list[PersonMetadata] = Field(default_factory=list)
    social_people: list[SocialPersonMetadata] = Field(default_factory=list)
    battery: int | None = None  # this might be redundant
