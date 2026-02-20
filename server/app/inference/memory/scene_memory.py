import logging
import re
import threading
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

    def __init__(
        self,
        memory_max_age_seconds: int = 60,
        memory_max_objects: int = 200,
        memory_max_relations: int = 500,
    ):
        logger.info("Initializing SceneMemory")
        self.tracks: dict[int, TrackedObject] = {}
        self.next_id = 1
        self.objects_state: dict[int, TrackedObjectState] = {}
        self.relations_state: dict[tuple[int, str, int], Relationship] = {}
        self.memory_max_age_seconds = memory_max_age_seconds
        self.memory_max_objects = memory_max_objects
        self.memory_max_relations = memory_max_relations
        self._lock = threading.Lock()

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

        with self._lock:
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
                        source="tracked",
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

            self._prune_memory_locked()

        return detections

    def _prune_memory_locked(self):
        now = time.time()
        cutoff = now - self.memory_max_age_seconds

        # Remove stale objects
        stale_ids = [
            obj_id
            for obj_id, obj in self.objects_state.items()
            if obj.last_seen < cutoff
        ]
        for obj_id in stale_ids:
            del self.objects_state[obj_id]

        # Remove relations referencing stale objects or stale relations
        stale_rel_keys = []
        for key, rel in self.relations_state.items():
            if rel.last_seen < cutoff:
                stale_rel_keys.append(key)
                continue
            if key[0] in stale_ids or key[2] in stale_ids:
                stale_rel_keys.append(key)
        for key in stale_rel_keys:
            del self.relations_state[key]

        # Enforce max objects
        if len(self.objects_state) > self.memory_max_objects:
            sorted_objs = sorted(self.objects_state.values(), key=lambda o: o.last_seen)
            to_remove = sorted_objs[: len(self.objects_state) - self.memory_max_objects]
            for obj in to_remove:
                self.objects_state.pop(obj.id, None)

        # Enforce max relations
        if len(self.relations_state) > self.memory_max_relations:
            sorted_rels = sorted(
                self.relations_state.values(), key=lambda r: r.last_seen
            )
            to_remove = sorted_rels[
                : len(self.relations_state) - self.memory_max_relations
            ]
            for rel in to_remove:
                key = (rel.subject_id, rel.predicate, rel.object_id)
                self.relations_state.pop(key, None)

    def prune_memory(self):
        with self._lock:
            self._prune_memory_locked()

    def set_limits(self, max_age_seconds: int, max_objects: int, max_relations: int):
        self.memory_max_age_seconds = max_age_seconds
        self.memory_max_objects = max_objects
        self.memory_max_relations = max_relations

    @staticmethod
    def _parse_id(value: str) -> int | None:
        try:
            return int(value)
        except Exception:
            match = re.search(r"(\d+)$", str(value))
            return int(match.group(1)) if match else None

    def update_scene_graph(self, scene_graph: SceneGraph):
        now = time.time()
        logger.info(
            f"Updating SceneMemory with a scene graph: {scene_graph.no_label_edges}"
        )
        with self._lock:
            for edge in scene_graph.no_label_edges:
                sub = self._parse_id(edge.sub)
                obj = self._parse_id(edge.obj)
                if sub is None or obj is None:
                    logger.debug(f"Skipping non-numeric edge: {edge}")
                    continue
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
        with self._lock:
            now = time.time()
            logger.info("Returning scene state from all memory")
            return SceneState(
                objects=list(self.objects_state.values()),
                relationships=list(self.relations_state.values()),
                timestamp=now,
            )

    def upsert_scene_state(self, state: SceneState):
        """Merge external SceneState into memory (for manual injections)."""
        now = time.time()
        with self._lock:
            for obj in state.objects:
                current = self.objects_state.get(obj.id)
                if current is None:
                    obj.source = obj.source or "manual"
                    self.objects_state[obj.id] = obj
                else:
                    current.label = obj.label or current.label
                    current.bbox = obj.bbox or current.bbox
                    current.attributes = list(
                        {*(current.attributes or []), *(obj.attributes or [])}
                    )
                    current.last_seen = obj.last_seen or now
                    current.hits = max(current.hits, obj.hits or 1)

            for rel in state.relationships:
                key = (rel.subject_id, rel.predicate, rel.object_id)
                existing = self.relations_state.get(key)
                if existing is None:
                    self.relations_state[key] = rel
                else:
                    existing.last_seen = rel.last_seen or now
                    existing.count = max(existing.count, rel.count or 1)

            if self.objects_state:
                max_id = max(self.objects_state.keys())
                self.next_id = max(self.next_id, max_id + 1)

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
