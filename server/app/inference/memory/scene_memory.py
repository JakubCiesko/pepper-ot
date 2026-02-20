from PIL import Image

from ..tracking.associator import Associator
from ..tracking.embeddings import FeatureExtractor
from ..types import DetectionObject
from ..types import TrackedObject


class SceneMemory:
    """Manages the lifecycle of objects in the robot's world."""

    def __init__(self):
        self.tracks: dict[int, TrackedObject] = {}
        self.next_id = 1

        # Dependencies
        self.extractor = FeatureExtractor()
        self.associator = Associator()

    def update(self, image: Image.Image, detections: list[DetectionObject]):
        """
        Main pipeline step:
        1. Extract Embeddings
        2. Match to History
        3. Update IDs in DetectionObjects
        """
        if not detections:
            return detections

        # 1. Extract Embeddings
        embeddings = self.extractor.extract(image, detections)

        # 2. Match
        active_tracks_list = list(self.tracks.values())
        matches, un_tracks, un_dets = self.associator.match(
            active_tracks_list, detections, embeddings
        )

        # 3. Update Matched Tracks
        for t_idx, d_idx in matches:
            track = active_tracks_list[t_idx]
            det = detections[d_idx]
            emb = embeddings[d_idx]

            # Update Track State
            track.update(det, emb)

            # ASSIGN ID TO DETECTION (Critical for SoM!)
            det.object_id = track.id

        # 4. Create New Tracks
        for d_idx in un_dets:
            det = detections[d_idx]
            emb = embeddings[d_idx]

            new_track = TrackedObject(
                id=self.next_id,
                label=det.label,
                embedding=emb,
                bbox=det.bbox,
                confidence=det.confidence,
            )
            self.tracks[self.next_id] = new_track

            # Assign ID
            det.object_id = self.next_id
            self.next_id += 1

        # 5. Prune (Simple logic for notebook)
        # In a real loop, you'd increment 'frames_since_seen' for un_tracks indices
        # and delete if > threshold.

        return detections

    def snapshot(self) -> list[dict]:
        """Return a lightweight view of the current tracked objects."""
        return [
            {
                "id": track.id,
                "label": track.label,
                "bbox": track.bbox,
                "confidence": track.confidence,
                "last_seen": track.last_seen,
                "first_seen": track.first_seen,
                "hits": track.hits,
            }
            for track in self.tracks.values()
        ]
