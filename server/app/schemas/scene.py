from pydantic import BaseModel
from pydantic import Field


class DetectionObject(BaseModel):
    label: str = Field(..., description="Object label")
    confidence: float = Field(..., description="Detection confidence")
    bbox: list[float] = Field(..., description="[x1, y1, x2, y2]")
    object_id: int | None = Field(None, description="Persistent tracking ID")


class DetectionResponse(BaseModel):
    objects: list[DetectionObject] | list[dict]
    timestamp: float
    image_width: int
    image_height: int


class Relationship(BaseModel):
    subject_id: int
    predicate: str
    object_id: int
    first_seen: float
    last_seen: float
    count: int = 1


class TrackedObjectState(BaseModel):
    id: int
    label: str
    status: str
    attributes: list[str] = Field(default_factory=list)
    first_seen: float
    last_seen: float
    hits: int = 1
    bbox: list[float]


class SceneState(BaseModel):
    objects: list[TrackedObjectState]
    relationships: list[Relationship]
    timestamp: float
