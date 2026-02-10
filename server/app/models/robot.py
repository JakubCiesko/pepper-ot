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
    people: list[PersonMetadata] = Field(default_factory=list)
    battery: int | None = None  # this might be redundant
