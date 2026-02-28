import logging
import math
import re
import threading
import time

from PIL import Image

from app.inference.tracking.associator import Associator
from app.inference.tracking.embeddings import FeatureExtractor
from app.inference.types import DetectionObject
from app.inference.types import SceneGraph
from app.inference.types import TrackedObject
from app.schemas.robot import RobotMetadata
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

    def update(
        self,
        image: Image.Image,
        detections: list[DetectionObject],
        robot_metadata: RobotMetadata | None = None,
        fusion_config=None,
    ):
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

            # 5. Fuse PeoplePerception into detections
            fused_detections = self._fuse_people_perception(
                detections, robot_metadata, image, fusion_config
            )

            # 6. Update persistent object state
            now = time.time()
            for det in fused_detections:
                if det.object_id is None:
                    continue
                bearing = self._compute_bearing(det, robot_metadata, image)
                current = self.objects_state.get(det.object_id)
                if current is None:
                    self.objects_state[det.object_id] = TrackedObjectState(
                        id=det.object_id,
                        label=det.label,
                        status="active",
                        source="tracked",
                        attributes=[],
                        bearing_yaw=bearing[0] if bearing else None,
                        bearing_pitch=bearing[1] if bearing else None,
                        frame_id=robot_metadata.frame_id if robot_metadata else None,
                        scan_id=robot_metadata.scan_id if robot_metadata else None,
                        first_seen=now,
                        last_seen=now,
                        hits=1,
                        bbox=det.bbox,
                    )
                else:
                    current.label = det.label
                    current.bbox = det.bbox
                    if bearing:
                        current.bearing_yaw, current.bearing_pitch = bearing
                    if robot_metadata:
                        current.frame_id = robot_metadata.frame_id or current.frame_id
                        current.scan_id = robot_metadata.scan_id or current.scan_id
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

    def _compute_bearing(
        self,
        det: DetectionObject,
        robot_metadata: RobotMetadata | None,
        image: Image.Image,
    ) -> tuple[float, float] | None:
        if robot_metadata is None:
            return None
        if robot_metadata.camera_hfov is None or robot_metadata.camera_vfov is None:
            return None
        width = robot_metadata.image_width or image.width
        height = robot_metadata.image_height or image.height
        x1, y1, x2, y2 = det.bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        yaw_rel = (0.5 - cx / width) * math.radians(robot_metadata.camera_hfov)
        pitch_rel = (0.5 - cy / height) * math.radians(robot_metadata.camera_vfov)
        base_yaw = (robot_metadata.body_yaw or 0.0) + robot_metadata.head_yaw
        base_pitch = robot_metadata.head_pitch
        return base_yaw + yaw_rel, base_pitch + pitch_rel

    def _fuse_people_perception(
        self,
        detections: list[DetectionObject],
        robot_metadata: RobotMetadata | None,
        image: Image.Image,
        fusion_config,
    ) -> list[DetectionObject]:
        if robot_metadata is None or not robot_metadata.people:
            return detections

        persons = [
            d for d in detections if d.label == "person" and d.object_id is not None
        ]
        others = [d for d in detections if d.label != "person" or d.object_id is None]

        width = robot_metadata.image_width or image.width
        height = robot_metadata.image_height or image.height
        hfov = robot_metadata.camera_hfov
        vfov = robot_metadata.camera_vfov
        if hfov is None or vfov is None:
            return detections

        match_thresh = getattr(fusion_config, "person_bbox_match_threshold_px", 10.0)
        base_px = getattr(fusion_config, "estimated_person_bbox_base_px", 80.0)
        min_px = getattr(fusion_config, "estimated_person_bbox_min_px", 40.0)
        max_px = getattr(fusion_config, "estimated_person_bbox_max_px", 200.0)

        base_yaw = (robot_metadata.body_yaw or 0.0) + robot_metadata.head_yaw
        base_pitch = robot_metadata.head_pitch

        def to_pixel(yaw, pitch):
            x = (0.5 - (yaw - base_yaw) / math.radians(hfov)) * width
            y = (0.5 - (pitch - base_pitch) / math.radians(vfov)) * height
            return x, y

        fused = []
        used_ids = set()
        for person in robot_metadata.people:
            px, py = to_pixel(person.yaw, person.pitch)
            matched = None
            for det in persons:
                x1, y1, x2, y2 = det.bbox
                if (x1 - match_thresh) <= px <= (x2 + match_thresh) and (
                    y1 - match_thresh
                ) <= py <= (y2 + match_thresh):
                    matched = det
                    break
            if matched:
                matched.confidence = max(matched.confidence, 1.0)
                used_ids.add(matched.object_id)
                fused.append(matched)
                continue

            scale = base_px / max(person.distance, 0.3)
            size = max(min(scale, max_px), min_px)
            x1 = max(0.0, px - size / 2)
            y1 = max(0.0, py - size / 2)
            x2 = min(width, px + size / 2)
            y2 = min(height, py + size / 2)

            det = DetectionObject(
                class_id=-1,
                label="person",
                confidence=1.0,
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                object_id=None,
            )
            fused.append(det)

        remaining = [p for p in persons if p.object_id not in used_ids]
        return fused + remaining + others

    @staticmethod
    def _parse_id(value: str) -> int | None:
        try:
            return int(value)
        except Exception:
            match = re.search(r"(\d+)$", str(value))
            return int(match.group(1)) if match else None

    def update_scene_graph(self, scene_graph: SceneGraph):
        if scene_graph is None:
            return
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
