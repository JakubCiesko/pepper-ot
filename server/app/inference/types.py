from app.models.robot import RobotMetadata
import numpy as np
from numpy.typing import NDArray
from PIL import Image


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
        self.timestamp = metadata.head_yaw  # Example usage
