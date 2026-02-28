import logging
import math
import re
import time

from PIL import Image

from app.inference.types import DetectionObject
from app.inference.types import SceneGraph
from app.inference.types import TrackedObject
from app.schemas.robot import RobotMetadata
from app.schemas.scene import Relationship
from app.schemas.scene import SceneState
from app.schemas.scene import TrackedObjectState

logger = logging.getLogger(__name__)


class SceneMemoryStore:
    """State container and mutation helpers for SceneMemory."""

    def __init__(
        self,
        memory_max_age_seconds: int = 60,
        memory_max_objects: int = 200,
        memory_max_relations: int = 500,
    ):
        self.tracks: dict[int, TrackedObject] = {}
        self.next_id = 1
        self.objects_state: dict[int, TrackedObjectState] = {}
        self.relations_state: dict[tuple[int, str, int], Relationship] = {}
        self.set_limits(
            max_age_seconds=memory_max_age_seconds,
            max_objects=memory_max_objects,
            max_relations=memory_max_relations,
        )

    def set_limits(self, max_age_seconds: int, max_objects: int, max_relations: int):
        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be > 0")
        if max_objects <= 0:
            raise ValueError("max_objects must be > 0")
        if max_relations <= 0:
            raise ValueError("max_relations must be > 0")
        self.memory_max_age_seconds = max_age_seconds
        self.memory_max_objects = max_objects
        self.memory_max_relations = max_relations

    def reset(self):
        self.tracks.clear()
        self.objects_state.clear()
        self.relations_state.clear()
        self.next_id = 1

    def insert_object(self, obj: TrackedObjectState):
        if obj.id in self.objects_state:
            raise ValueError(f"Object with id={obj.id} already exists")
        self.objects_state[obj.id] = obj.model_copy(deep=True)
        self.next_id = max(self.next_id, obj.id + 1)

    def patch_object(self, object_id: int, updates: dict) -> TrackedObjectState:
        current = self.objects_state.get(object_id)
        if current is None:
            raise KeyError(f"Object with id={object_id} does not exist")
        for field in (
            "label",
            "status",
            "source",
            "attributes",
            "bearing_yaw",
            "bearing_pitch",
            "frame_id",
            "scan_id",
            "first_seen",
            "last_seen",
            "hits",
            "bbox",
        ):
            if field in updates:
                setattr(current, field, updates[field])

        # Keep track representation in sync for shared fields.
        track = self.tracks.get(object_id)
        if track is not None:
            if "label" in updates:
                track.label = current.label
            if "bbox" in updates:
                track.bbox = current.bbox
            if "last_seen" in updates:
                track.last_seen = current.last_seen
            if "first_seen" in updates:
                track.first_seen = current.first_seen
            if "hits" in updates:
                track.hits = current.hits
        return current

    def delete_object(self, object_id: int, cascade_relations: bool = True) -> bool:
        removed = self.objects_state.pop(object_id, None)
        if removed is None:
            return False
        self.tracks.pop(object_id, None)
        if cascade_relations:
            keys_to_remove = [
                key
                for key in self.relations_state
                if key[0] == object_id or key[2] == object_id
            ]
            for key in keys_to_remove:
                self.relations_state.pop(key, None)
        return True

    def insert_relation(self, rel: Relationship):
        if rel.subject_id not in self.objects_state:
            raise ValueError(f"Subject object id={rel.subject_id} does not exist")
        if rel.object_id not in self.objects_state:
            raise ValueError(f"Object object id={rel.object_id} does not exist")
        key = (rel.subject_id, rel.predicate, rel.object_id)
        if key in self.relations_state:
            raise ValueError("Relationship already exists")
        self.relations_state[key] = rel.model_copy(deep=True)

    def patch_relation(
        self,
        subject_id: int,
        predicate: str,
        object_id: int,
        updates: dict,
    ) -> Relationship:
        old_key = (subject_id, predicate, object_id)
        current = self.relations_state.get(old_key)
        if current is None:
            raise KeyError(
                f"Relationship ({subject_id}, {predicate}, {object_id}) does not exist"
            )
        new_subject_id = updates.get("subject_id", current.subject_id)
        new_predicate = updates.get("predicate", current.predicate)
        new_object_id = updates.get("object_id", current.object_id)

        if new_subject_id not in self.objects_state:
            raise ValueError(f"Subject object id={new_subject_id} does not exist")
        if new_object_id not in self.objects_state:
            raise ValueError(f"Object object id={new_object_id} does not exist")

        new_key = (new_subject_id, new_predicate, new_object_id)
        if new_key != old_key and new_key in self.relations_state:
            raise ValueError(
                f"Relationship ({new_subject_id}, {new_predicate}, {new_object_id}) already exists"
            )

        self.relations_state.pop(old_key, None)
        current.subject_id = new_subject_id
        current.predicate = new_predicate
        current.object_id = new_object_id

        for field in ("first_seen", "last_seen", "count"):
            if field in updates:
                setattr(current, field, updates[field])
        self.relations_state[new_key] = current
        return current

    def delete_relation(self, subject_id: int, predicate: str, object_id: int) -> bool:
        key = (subject_id, predicate, object_id)
        return self.relations_state.pop(key, None) is not None

    def create_track(self, det: DetectionObject, embedding) -> int:
        object_id = self.next_id
        self.tracks[object_id] = TrackedObject(
            id=object_id,
            label=det.label,
            embedding=embedding,
            bbox=det.bbox,
            confidence=det.confidence,
        )
        self.next_id += 1
        return object_id

    def age_unmatched_tracks(
        self, unmatched_track_indices: list[int], active_tracks: list[TrackedObject]
    ):
        for t_idx in unmatched_track_indices:
            if t_idx < 0 or t_idx >= len(active_tracks):
                continue
            track = active_tracks[t_idx]
            track.frames_since_seen += 1

    def update_objects_from_detections(
        self,
        detections: list[DetectionObject],
        robot_metadata: RobotMetadata | None,
        image: Image.Image,
    ):
        now = time.time()
        for det in detections:
            if det.object_id is None:
                continue
            bearing = self.compute_bearing(det, robot_metadata, image)
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

    def prune_memory(self):
        now = time.time()
        cutoff = now - self.memory_max_age_seconds

        stale_object_ids = {
            obj_id
            for obj_id, obj in self.objects_state.items()
            if obj.last_seen < cutoff
        }

        if len(self.objects_state) > self.memory_max_objects:
            sorted_objs = sorted(self.objects_state.values(), key=lambda o: o.last_seen)
            overflow = len(self.objects_state) - self.memory_max_objects
            stale_object_ids.update(obj.id for obj in sorted_objs[:overflow])

        for obj_id in stale_object_ids:
            self.objects_state.pop(obj_id, None)

        stale_relation_keys = []
        for key, rel in self.relations_state.items():
            if rel.last_seen < cutoff:
                stale_relation_keys.append(key)
                continue
            if key[0] in stale_object_ids or key[2] in stale_object_ids:
                stale_relation_keys.append(key)
        for key in stale_relation_keys:
            self.relations_state.pop(key, None)

        if len(self.relations_state) > self.memory_max_relations:
            sorted_rels = sorted(
                self.relations_state.values(), key=lambda r: r.last_seen
            )
            overflow = len(self.relations_state) - self.memory_max_relations
            for rel in sorted_rels[:overflow]:
                key = (rel.subject_id, rel.predicate, rel.object_id)
                self.relations_state.pop(key, None)

        stale_track_ids = {
            track_id
            for track_id, track in self.tracks.items()
            if track.last_seen < cutoff
        }
        stale_track_ids.update(
            track_id for track_id in self.tracks if track_id not in self.objects_state
        )
        if len(self.tracks) > self.memory_max_objects:
            sorted_tracks = sorted(self.tracks.values(), key=lambda t: t.last_seen)
            overflow = len(self.tracks) - self.memory_max_objects
            stale_track_ids.update(track.id for track in sorted_tracks[:overflow])
        for track_id in stale_track_ids:
            self.tracks.pop(track_id, None)

    @staticmethod
    def compute_bearing(
        det: DetectionObject,
        robot_metadata: RobotMetadata | None,
        image: Image.Image,
    ) -> tuple[float, float] | None:
        if robot_metadata is None:
            return None
        if robot_metadata.camera_hfov is None or robot_metadata.camera_vfov is None:
            return None
        if len(det.bbox) != 4:
            return None
        width = robot_metadata.image_width or image.width
        height = robot_metadata.image_height or image.height
        if width <= 0 or height <= 0:
            return None
        x1, y1, x2, y2 = det.bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        yaw_rel = (0.5 - cx / width) * math.radians(robot_metadata.camera_hfov)
        pitch_rel = (0.5 - cy / height) * math.radians(robot_metadata.camera_vfov)
        base_yaw = (robot_metadata.body_yaw or 0.0) + robot_metadata.head_yaw
        base_pitch = robot_metadata.head_pitch
        return base_yaw + yaw_rel, base_pitch + pitch_rel

    def fuse_people_perception(
        self,
        detections: list[DetectionObject],
        robot_metadata: RobotMetadata | None,
        image: Image.Image,
        fusion_config,
    ) -> list[DetectionObject]:
        if robot_metadata is None or not robot_metadata.people:
            return detections

        width = robot_metadata.image_width or image.width
        height = robot_metadata.image_height or image.height
        hfov = robot_metadata.camera_hfov
        vfov = robot_metadata.camera_vfov
        if hfov is None or vfov is None or width <= 0 or height <= 0:
            return detections

        persons = [
            d for d in detections if d.label == "person" and d.object_id is not None
        ]
        others = [d for d in detections if d.label != "person" or d.object_id is None]

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
                if len(det.bbox) != 4:
                    continue
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
            if x2 <= x1 or y2 <= y1:
                continue

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
        edges = getattr(scene_graph, "no_label_edges", None) or []
        for edge in edges:
            sub = self._parse_id(getattr(edge, "sub", ""))
            obj = self._parse_id(getattr(edge, "obj", ""))
            rel = getattr(edge, "rel", None)
            if sub is None or obj is None or not rel:
                continue
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
        return SceneState(
            objects=list(self.objects_state.values()),
            relationships=list(self.relations_state.values()),
            timestamp=time.time(),
        )

    def upsert_scene_state(self, state: SceneState):
        now = time.time()
        for obj in state.objects:
            current = self.objects_state.get(obj.id)
            if current is None:
                self.objects_state[obj.id] = obj.model_copy(deep=True)
                if not self.objects_state[obj.id].source:
                    self.objects_state[obj.id].source = "manual"
            else:
                current.label = obj.label or current.label
                current.status = obj.status or current.status
                current.source = obj.source or current.source
                current.bbox = obj.bbox or current.bbox
                current.attributes = list(
                    {*(current.attributes or []), *(obj.attributes or [])}
                )
                current.bearing_yaw = (
                    obj.bearing_yaw
                    if obj.bearing_yaw is not None
                    else current.bearing_yaw
                )
                current.bearing_pitch = (
                    obj.bearing_pitch
                    if obj.bearing_pitch is not None
                    else current.bearing_pitch
                )
                current.frame_id = obj.frame_id or current.frame_id
                current.scan_id = obj.scan_id or current.scan_id
                current.first_seen = min(current.first_seen, obj.first_seen or now)
                current.last_seen = max(current.last_seen, obj.last_seen or now)
                current.hits = max(current.hits, obj.hits or 1)

        for rel in state.relationships:
            key = (rel.subject_id, rel.predicate, rel.object_id)
            existing = self.relations_state.get(key)
            if existing is None:
                self.relations_state[key] = rel.model_copy(deep=True)
            else:
                existing.first_seen = min(existing.first_seen, rel.first_seen or now)
                existing.last_seen = max(existing.last_seen, rel.last_seen or now)
                existing.count = max(existing.count, rel.count or 1)

        if self.objects_state:
            max_id = max(self.objects_state.keys())
            self.next_id = max(self.next_id, max_id + 1)

    def snapshot(self) -> list[dict]:
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
