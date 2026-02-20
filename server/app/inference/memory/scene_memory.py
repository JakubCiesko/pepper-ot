import logging
import time

from PIL import Image

from app.inference.tracking.associator import Associator
from app.inference.tracking.embeddings import FeatureExtractor
from app.inference.types import DetectionObject
from app.inference.types import SceneGraph
from app.inference.types import TrackedObject
from app.schemas.scene import Relationship
from app.schemas.scene import SceneState
from app.schemas.scene import TrackedObjectState

logger = logging.getLogger(__file__)


class SceneMemory:
    """Manages the lifecycle of objects in the robot's world."""

    def __init__(self):
        logger.info("Initializing SceneMemory")
        self.tracks: dict[int, TrackedObject] = {}
        self.next_id = 1
        self.objects_state: dict[int, TrackedObjectState] = {}
        self.relations_state: dict[tuple[int, str, int], Relationship] = {}

        # Dependencies
        self.extractor = FeatureExtractor()
        self.associator = Associator()
        logger.info("SceneMemory initialized")

    def update(self, image: Image.Image, detections: list[DetectionObject]):
        """
        Main pipeline step:
        1. Extract Embeddings
        2. Match to History
        3. Update IDs in DetectionObjects
        """
        logger.info(f"Updating SceneMemory with {len(detections)} detections")
        if not detections:
            logger.info("No detection provided, SceneMemory not updated.")
            return detections

        # 1. Extract Embeddings
        logger.debug("Passing image and detection to embedding extractor")
        embeddings = self.extractor.extract(image, detections)

        # 2. Match
        logger.debug("Passing embeddings and detections to associator")
        active_tracks_list = list(self.tracks.values())
        matches, un_tracks, un_dets = self.associator.match(
            active_tracks_list, detections, embeddings
        )

        # 3. Update Matched Tracks
        logger.debug(f"Updating {len(matches)} Matched Tracks")
        for t_idx, d_idx in matches:
            track = active_tracks_list[t_idx]
            det = detections[d_idx]
            emb = embeddings[d_idx]

            # Update Track State
            track.update(det, emb)

            # ASSIGN ID TO DETECTION (Critical for SoM!)
            det.object_id = track.id

        # 4. Create New Tracks
        logger.debug(f"Creating {len(un_dets)} new tracks (unmatched objects)")
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

        # 5. Update persistent object state
        now = time.time()
        for det in detections:
            if det.object_id is None:
                continue
            current = self.objects_state.get(det.object_id)
            if current is None:
                self.objects_state[det.object_id] = TrackedObjectState(
                    id=det.object_id,
                    label=det.label,
                    status="active",
                    attributes=[],
                    first_seen=now,
                    last_seen=now,
                    hits=1,
                    bbox=det.bbox,
                )
            else:
                current.label = det.label
                current.bbox = det.bbox
                current.last_seen = now
                current.hits += 1

        # TODO: PRUNING OLD THINGS FROM MEMORY SETTABLE TIME DURATION
        # 6. Prune (Simple logic for notebook)
        self.prune_memory()
        # In a real loop, you'd increment 'frames_since_seen' for un_tracks indices
        # and delete if > threshold.

        return detections

    def prune_memory(self):
        pass

    def update_scene_graph(self, scene_graph: SceneGraph):
        now = time.time()
        logger.info(
            f"Updating SceneMemory with a scene graph: {scene_graph.no_label_edges}"
        )
        for (
            edge
        ) in (
            scene_graph.no_label_edges
        ):  # this is always only integers, never label_integer
            sub = int(edge.sub)
            obj = int(edge.obj)
            rel = edge.rel

            if sub == obj:
                state = self.objects_state.get(sub)
                if state and rel not in state.attributes:
                    state.attributes.append(rel)
                continue

            key = (sub, rel, obj)
            existing = self.relations_state.get(key)
            if existing is None:
                self.relations_state[key] = Relationship(
                    subject_id=sub,
                    predicate=rel,
                    object_id=obj,
                    first_seen=now,
                    last_seen=now,
                    count=1,
                )
            else:
                existing.last_seen = now
                existing.count += 1

    def scene_state(self) -> SceneState:
        now = time.time()
        logger.info("Returning scene state from all memory")
        return SceneState(
            objects=list(self.objects_state.values()),
            relationships=list(self.relations_state.values()),
            timestamp=now,
        )

    def snapshot(self) -> list[dict]:
        """Return a lightweight view of the current tracked objects."""
        logger.info("Returning latest scene state")
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
