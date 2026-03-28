from uuid import UUID
from uuid import uuid4

from pydantic import BaseModel
from pydantic import Field


class DetectionObject(BaseModel):
    label: str = Field(..., description="Object label")
    confidence: float = Field(..., description="Detection confidence")
    bbox: list[float] = Field(..., description="[x1, y1, x2, y2]")
    object_id: int | None = Field(None, description="Persistent tracking ID")


class DetectionResponse(BaseModel):
    id: UUID | str | int | None = Field(
        default_factory=uuid4, description="Persistent Detection Response ID"
    )
    objects: list[DetectionObject] | list[dict]
    timestamp: float
    image_width: int
    image_height: int
    caption: str | None = None
    caption_provider: str | None = None
    caption_model_id: str | None = None


class DetectFormRequest(BaseModel):
    metadata: str | None = Field(
        default=None,
        description="JSON string with RobotMetadata payload",
    )
    publish: bool = Field(
        default=True,
        description="Broadcast detection event to dashboard/ws",
    )
    resize_image: bool = Field(
        default=True,
        description="Resize image before running detection to mitigate GPU load",
    )
