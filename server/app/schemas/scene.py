from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel
from pydantic import Field


def time_ago(timestamp: float) -> str:
    tz = ZoneInfo("Europe/Bratislava")

    now = datetime.now(tz)
    last_seen = datetime.fromtimestamp(timestamp, tz=tz)
    diff = now - last_seen

    seconds = int(diff.total_seconds())

    if seconds < 10:
        return "just now"
    if seconds < 60:
        return f"{seconds} seconds ago"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = seconds // 86400
        return f"{days} day{'s' if days != 1 else ''} ago"


# This serves for structured output for openai and gemini TODO: maybe just one class.
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
    pepper_person_id: int | None = None
    robot_distance: float | None = None
    robot_engagement_zone: int | None = None
    robot_last_seen_ts: float | None = None
    bearing_yaw: float | None = None
    bearing_pitch: float | None = None
    frame_id: str | None = None
    scan_id: str | None = None
    first_seen: float
    last_seen: float
    hits: int = 1
    bbox: list[float]

    def last_seen_human_format(self):
        return time_ago(self.last_seen)


class SceneCaptionState(BaseModel):
    id: str
    text: str
    provider: str | None = None
    model_id: str | None = None
    source: str = "pipeline_caption"
    frame_id: str | None = None
    scan_id: str | None = None
    first_seen: float
    last_seen: float
    count: int = 1


class SceneState(BaseModel):
    objects: list[TrackedObjectState]
    relationships: list[Relationship]
    captions: list[SceneCaptionState] = Field(default_factory=list)
    timestamp: float

# TODO: look in creation in memoy_graph_render.py ask yourself whther
# to send over relations, attributes, and their counts.
class MemorySummary(BaseModel):
    timestamp: float
    labels: list[str] = Field(default_factory=list)
    label_counts: dict[str, int] = Field(default_factory=dict)
    scene_graph: list[SceneGraphRelation] = Field(default_factory=list)
    graph_svg: str | None = None
    pregenerated_qa: list[dict[str, str]] | None = None
