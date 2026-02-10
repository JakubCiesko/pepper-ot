from pydantic import BaseModel
from pydantic import Field


class DetectionObject(BaseModel):
    label: str = Field(..., description="Object label")
    confidence: float = Field(..., description="Detection confidence")
    bbox: list[float] = Field(..., description="[x1, y1, x2, y2]")
    object_id: int | None = Field(None, description="Persistent tracking ID")


class DetectionResponse(BaseModel):
    objects: list[DetectionObject]
    timestamp: float
    image_width: int
    image_height: int


class Relationship(BaseModel):
    subject_id: int
    predicate: str
    object_id: int


class TrackedObjectState(BaseModel):
    id: int
    label: str
    status: str
    last_seen: float
    bbox: list[float]


class SceneState(BaseModel):
    objects: list[TrackedObjectState]
    relationships: list[Relationship]
    timestamp: float
