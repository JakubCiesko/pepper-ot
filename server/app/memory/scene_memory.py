from enum import Enum
import logging
import time

from app.inference.types import InternalDetection
import numpy as np

logger = logging.getLogger(__name__)


class ObjectStatus(Enum):
    TRACKED = "tracked"
    DORMANT = "dormant"
    LOST = "lost"


class TrackedObject:
    """The persistent identity of a detected entity."""

    def __init__(self, obj_id: int, det: InternalDetection):
        self.id = obj_id
        self.label = det.label
        self.bbox = det.bbox
        self.embedding = det.embedding  # Normalized NDArray
        self.confidence = det.confidence

        self.status = ObjectStatus.TRACKED
        self.hits = 1
        self.frames_since_seen = 0
        self.last_seen = time.time()
        self.first_seen = self.last_seen

    def update(self, det: InternalDetection):
        self.bbox = det.bbox
        self.confidence = det.confidence

        # Smooth embedding (moving average)
        self.embedding = 0.9 * self.embedding + 0.1 * det.embedding
        self.embedding /= np.linalg.norm(self.embedding)  # Re-normalize

        self.last_seen = time.time()
        self.hits += 1
        self.frames_since_seen = 0
        self.status = ObjectStatus.TRACKED

    def predict(self):
        self.frames_since_seen += 1
        self.status = ObjectStatus.DORMANT


class SceneMemory:
    """Orchestrates the lifecycle of the World Model."""

    def __init__(self, max_dormant_frames: int = 30):
        self.tracks: dict[int, TrackedObject] = {}
        self.next_id = 1
        self.max_dormant_frames = max_dormant_frames

    def add_new_track(self, det: InternalDetection):
        new_track = TrackedObject(self.next_id, det)
        self.tracks[self.next_id] = new_track
        self.next_id += 1
        return new_track

    def prune(self):
        to_delete = [
            tid
            for tid, t in self.tracks.items()
            if t.frames_since_seen > self.max_dormant_frames
        ]
        for tid in to_delete:
            del self.tracks[tid]
        if to_delete:
            logger.info(f"Pruned {len(to_delete)} objects from memory.")
