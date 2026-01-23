from pydantic import BaseModel
from pydantic import Field


class DetectionObject(BaseModel):
    label: str = Field(..., description="Detected object label")
    confidence: float = Field(
        ..., gt=0, lt=1, description="Confidence score of detection"
    )
    bbox: list[float] = Field(
        ..., description="Bounding box of object", min_length=4, max_length=4
    )


class DetectionResponse(BaseModel):
    objects: list[DetectionObject] = Field(
        ..., description="List of detected objects of type DetectionObject"
    )
