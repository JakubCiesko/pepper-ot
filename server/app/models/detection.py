from pydantic import BaseModel
from pydantic import Field


class DetectionObject(BaseModel):
    label: str = Field(..., description="Detected object label")
    confidence: float = Field(..., description="Confidence score")
    bbox: list[float] = Field(
        ..., description="[x1, y1, x2, y2]", min_length=4, max_length=4
    )
    object_id: int | None = Field(None, description="Persistent ID from tracking")


class DetectionResponse(BaseModel):
    objects: list[DetectionObject]
    timestamp: float
    image_width: int
    image_height: int
