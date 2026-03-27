from pydantic import BaseModel
from pydantic import Field


# This serves for structured output for openai and gemini
class SceneGraphRelation(BaseModel):
    sub: str
    rel: str
    obj: str


class SceneGraphStructuredResponse(BaseModel):
    relationships: list[SceneGraphRelation] = Field(default_factory=list)


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
    source: str = "tracked"
    attributes: list[str] = Field(default_factory=list)
    bearing_yaw: float | None = None
    bearing_pitch: float | None = None
    frame_id: str | None = None
    scan_id: str | None = None
    first_seen: float
    last_seen: float
    hits: int = 1
    bbox: list[float]


class SceneState(BaseModel):
    objects: list[TrackedObjectState]
    relationships: list[Relationship]
    timestamp: float
