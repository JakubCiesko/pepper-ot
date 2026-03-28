from dataclasses import dataclass
from dataclasses import field
import re
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel
from pydantic import Field

from app.schemas.robot import RobotMetadata


class InferenceDetectionObject(BaseModel):
    class_id: int = Field(..., description="Class ID")
    label: str = Field(..., description="Object label")
    confidence: float = Field(..., description="Detection confidence")
    bbox: list[float] = Field(..., description="[x1, y1, x2, y2]")
    object_id: int | None = Field(None, description="Persistent tracking ID")


@dataclass
class TrackedObject:
    """The persistent identity of a detected entity."""

    id: int
    label: str
    embedding: np.ndarray
    bbox: list[float]
    confidence: float

    # State
    last_seen: float = field(default_factory=time.time)
    first_seen: float = field(default_factory=time.time)
    hits: int = 1
    frames_since_seen: int = 0

    @property
    def center(self):
        """Returns (x_center, y_center)"""
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    def update(self, det: InferenceDetectionObject, embedding: np.ndarray):
        """Update state with new observation."""
        self.bbox = det.bbox
        self.confidence = det.confidence

        # Smooth embedding (Exponential Moving Average) to stabilize identity
        # We give 10% weight to the new look, 90% to history
        self.embedding = 0.9 * self.embedding + 0.1 * embedding
        self.embedding /= np.linalg.norm(self.embedding)  # Re-normalize

        self.last_seen = time.time()
        self.hits += 1
        self.frames_since_seen = 0


class BoundingBox:
    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def iou(self, other: "BoundingBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union_area = self.area + other.area - inter_area
        return inter_area / union_area if union_area > 0 else 0


class InternalDetection:
    """Internal transient detection holding NumPy embeddings."""

    def __init__(
        self, label: str, bbox: BoundingBox, embedding: NDArray, confidence: float
    ):
        self.label = label
        self.bbox = bbox
        self.embedding = embedding  # Normalized NDArray
        self.confidence = confidence


class CameraModel:
    """Math for mapping pixels to angles (Yaw/Pitch)."""

    def __init__(
        self,
        h_fov: float = 57.2,
        v_fov: float = 44.3,
        width: int = 640,
        height: int = 480,
    ):
        self.h_fov = np.radians(h_fov)
        self.v_fov = np.radians(v_fov)
        self.width = width
        self.height = height

    def pixel_to_angles(self, x: float, y: float) -> tuple[float, float]:
        """Returns (yaw_rel, pitch_rel) in radians."""
        yaw = (0.5 - x / self.width) * self.h_fov
        pitch = (0.5 - y / self.height) * self.v_fov
        return yaw, pitch


class FrameContext:
    """Container for a single perception heartbeat."""

    def __init__(self, image: Image.Image, metadata: RobotMetadata):
        self.image = image
        self.metadata = metadata
        self.detections: list[InternalDetection] = []
        self.timestamp = time.time()  # Example usage


@dataclass
class SceneGraphEdge:
    sub: str
    rel: str
    obj: str

    def __hash__(self):
        return hash((self.sub, self.rel, self.obj))

    def __eq__(self, other):
        if not isinstance(other, SceneGraphEdge):
            return NotImplemented
        return (self.sub, self.rel, self.obj) == (other.sub, other.rel, other.obj)


@dataclass
class SceneGraph:
    """
    A structured scene graph object, returned by SceneGraphGenerator.generate.
    """

    edges: list[SceneGraphEdge] = field(default_factory=list)
    no_label_edges: list[SceneGraphEdge] = field(default_factory=list)
    raw: Any | None = None  # raw output from the VLM, useful for debugging

    def __post_init__(self):
        self.deduplicate()

    @staticmethod
    def _normalize_id(s: str) -> str:
        """Extract numeric ID from strings like 'cat_1' or return as is if already numeric"""
        match = re.search(r"(\d+)$", s)
        return match.group(1) if match else s

    @classmethod
    def from_list(cls, data: list[dict], raw: Any = None) -> "SceneGraph":
        edges = []
        for item in data:
            if all(k in item for k in ["sub", "rel", "obj"]):
                edges.append(
                    SceneGraphEdge(sub=item["sub"], rel=item["rel"], obj=item["obj"])
                )
            else:
                # fallback for unexpected structure
                edges.append(
                    SceneGraphEdge(
                        sub=str(item.get("sub", "")),
                        rel=str(item.get("rel", "")),
                        obj=str(item.get("obj", "")),
                    )
                )
        no_label_edges = [
            SceneGraphEdge(
                sub=cls._normalize_id(edge.sub),
                rel=edge.rel,
                obj=cls._normalize_id(edge.obj),
            )
            for edge in edges
        ]
        return cls(edges=edges, no_label_edges=no_label_edges, raw=raw)

    def deduplicate(self):
        self.edges = list(set(self.edges))
        self.no_label_edges = list(set(self.no_label_edges))

    def subjects(self) -> list[str]:
        return [edge.sub for edge in self.edges]

    def objects(self) -> list[str]:
        return [edge.obj for edge in self.edges]

    def predicates(self) -> list[str]:
        return [edge.rel for edge in self.edges]

    def attributes(self) -> list[str]:
        return [edge.rel for edge in self.edges if edge.obj == edge.sub]

    def as_dict(self) -> list[dict]:
        return [{"sub": e.sub, "rel": e.rel, "obj": e.obj} for e in self.edges]

    def __len__(self):
        return len(self.edges)

    def __add__(self, other: "SceneGraph") -> "SceneGraph":
        if not isinstance(other, SceneGraph):
            return NotImplemented
        merged = SceneGraph(
            edges=self.no_label_edges + other.no_label_edges,
            no_label_edges=self.no_label_edges + other.no_label_edges,
            raw=None,
        )
        return merged


@dataclass
class PipelineResult:
    """Holds the complete state of a processed frame."""

    raw_image: Image.Image
    som_image: np.ndarray | None  # The image with tags drawn on it
    detections: list[InferenceDetectionObject]  # List of objects with persistent IDs
    scene_graph: SceneGraph | None  # The semantic relationships
    metrics: dict[str, Any]
    executed_stages: list[str] = field(default_factory=list)
    caption: str | None = None
    caption_provider: str | None = None
    caption_model_id: str | None = None
