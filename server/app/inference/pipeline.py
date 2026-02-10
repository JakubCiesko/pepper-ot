import logging
import time

from app.inference.detection.service import DetectionService
from app.inference.tracking.associator import Associator
from app.inference.tracking.embeddings import FeatureExtractor
from app.memory.scene_memory import Detection
from app.memory.scene_memory import SceneMemory
from app.models.detection import DetectionObject
from app.models.detection import DetectionResponse
from PIL import Image

logger = logging.getLogger(__name__)


class InferencePipeline:
    """The orchestrator of the 'See, Track, Understand' logic."""

    def __init__(self):
        self.detector = DetectionService()
        self.extractor = FeatureExtractor()
        self.associator = Associator()
        self.memory = SceneMemory()

    def process_frame(self, image: Image.Image) -> DetectionResponse:
        w, h = image.size

        # 1. Detection
        raw_detections = self.detector.detect(image)
        if not raw_detections:
            self.memory.update([])  # Still update memory to mark objects as missing
            return self._build_response([], w, h)

        # 2. Embedding Extraction (Batch)
        crops = [image.crop(d.bbox) for d in raw_detections]
        embeddings = self.extractor.extract_batch(crops)

        # 3. Create Internal Detection Objects
        current_frame_detections = []
        for i, d in enumerate(raw_detections):
            rel_angle = (sum(d.bbox[::2]) / 2 / w) - 0.5  # (x1+x2)/2 / w - 0.5
            current_frame_detections.append(
                Detection(
                    label=d.label, bbox=d.bbox, embedding=embeddings[i], angle=rel_angle
                )
            )

        # 4. Data Association & Memory Update
        active_tracks = self.memory.get_active_tracks()
        matches, unmatched_tracks, unmatched_detections = self.associator.match(
            active_tracks, current_frame_detections
        )

        # Update matched tracks
        for track_idx, det_idx in matches:
            active_tracks[track_idx].update(current_frame_detections[det_idx])

        # Mark unmatched tracks as missing
        for track_idx in unmatched_tracks:
            active_tracks[track_idx].predict()

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self.memory.create_track(current_frame_detections[det_idx])

        # 5. Build Response
        # We only return objects currently in the frame
        final_objects = [
            DetectionObject(
                label=t.label,
                confidence=1.0,  # We can track confidence history later if needed
                bbox=t.bbox,
                object_id=t.id,
            )
            for t in self.memory.get_active_tracks()
            if t.frames_since_seen == 0
        ]
        self.memory.prune()
        return self._build_response(final_objects, w, h)

    @staticmethod
    def _build_response(
        objects: list[DetectionObject], w: int, h: int
    ) -> DetectionResponse:
        return DetectionResponse(
            objects=objects, timestamp=time.time(), image_width=w, image_height=h
        )
