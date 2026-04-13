from dataclasses import dataclass
from dataclasses import field
import logging
import time

from app.inference.memory.state_store.geometry import SceneMemoryStoreGeometryMixin
from app.inference.memory.state_store.objects import SceneMemoryStoreObjectsMixin
from app.inference.memory.state_store.relations import SceneMemoryStoreRelationsMixin
from app.inference.memory.state_store.social import SceneMemoryStoreSocialMixin
from app.inference.memory.state_store.tracks import SceneMemoryStoreTracksMixin
from app.inference.types import InferenceDetectionObject
from app.inference.types import TrackedObject
from app.schemas.scene import Relationship
from app.schemas.scene import SceneCaptionState
from app.schemas.scene import SceneState
from app.schemas.scene import TrackedObjectState

logger = logging.getLogger(__name__)


@dataclass
class PepperPersonBinding:
    pepper_person_id: int
    server_object_id: int
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    confidence: float = 1.0
    # consecutive_hits: int = 1
    misses: int = 0


class SceneMemoryStore(
    SceneMemoryStoreTracksMixin,
    SceneMemoryStoreGeometryMixin,
    SceneMemoryStoreSocialMixin,
    SceneMemoryStoreObjectsMixin,
    SceneMemoryStoreRelationsMixin,
):
    """State container and mutation helpers for SceneMemory."""

    def __init__(
        self,
        memory_max_age_seconds: int = 60,
        memory_max_objects: int = 200,
        memory_max_relations: int = 500,
        memory_max_captions: int = 100,
        caption_max_age_seconds: int = 600,
    ):
        self.tracks: dict[int, TrackedObject] = {}
        self.next_id = 1
        self.objects_state: dict[int, TrackedObjectState] = {}
        self.relations_state: dict[tuple[int, str, int], Relationship] = {}
        self.captions_state: dict[str, SceneCaptionState] = {}
        self.pepper_person_bindings: dict[int, PepperPersonBinding] = {}
        self._frame_server_to_pepper: dict[int, int] = {}
        self._pending_synthetic_pepper_by_detection: dict[int, int] = {}
        self.set_limits(
            max_age_seconds=memory_max_age_seconds,
            max_objects=memory_max_objects,
            max_relations=memory_max_relations,
            max_captions=memory_max_captions,
            caption_max_age_seconds=caption_max_age_seconds,
        )
        logger.info(
            "SceneMemoryStore initialized max_age=%s max_objects=%s max_relations=%s max_captions=%s caption_max_age=%s",
            memory_max_age_seconds,
            memory_max_objects,
            memory_max_relations,
            memory_max_captions,
            caption_max_age_seconds,
        )

    def scene_state(self) -> SceneState:
        scene_state = SceneState(
            objects=list(self.objects_state.values()),
            relationships=list(self.relations_state.values()),
            captions=list(self.captions_state.values()),
            timestamp=time.time(),
        )
        logger.debug(
            "SceneMemoryStore returning Current Scene State: %s",
            scene_state.model_dump_json(),
        )
        return scene_state

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
                current.pepper_person_id = (
                    obj.pepper_person_id
                    if obj.pepper_person_id is not None
                    else current.pepper_person_id
                )
                current.robot_distance = (
                    obj.robot_distance
                    if obj.robot_distance is not None
                    else current.robot_distance
                )
                current.robot_engagement_zone = (
                    obj.robot_engagement_zone
                    if obj.robot_engagement_zone is not None
                    else current.robot_engagement_zone
                )
                current.robot_last_seen_ts = (
                    obj.robot_last_seen_ts
                    if obj.robot_last_seen_ts is not None
                    else current.robot_last_seen_ts
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
        for caption in state.captions:
            existing = self.captions_state.get(caption.id)
            if existing is None:
                self.captions_state[caption.id] = caption.model_copy(deep=True)
            else:
                existing.text = caption.text or existing.text
                existing.provider = caption.provider or existing.provider
                existing.model_id = caption.model_id or existing.model_id
                existing.source = caption.source or existing.source
                existing.frame_id = caption.frame_id or existing.frame_id
                existing.scan_id = caption.scan_id or existing.scan_id
                existing.first_seen = min(
                    existing.first_seen, caption.first_seen or now
                )
                existing.last_seen = max(existing.last_seen, caption.last_seen or now)
                existing.count = max(existing.count, caption.count or 1)

        if self.objects_state:
            max_id = max(self.objects_state.keys())
            self.next_id = max(self.next_id, max_id + 1)
        logger.info(
            "Upserted scene state objects=%s relationships=%s captions=%s",
            len(state.objects),
            len(state.relationships),
            len(state.captions),
        )

    def clear_frame_pepper_state(self):
        self._frame_server_to_pepper.clear()
        self._pending_synthetic_pepper_by_detection.clear()

    def get_pepper_binding(self, pepper_person_id: int) -> PepperPersonBinding | None:
        return self.pepper_person_bindings.get(pepper_person_id)

    def get_bound_server_object_id(self, pepper_person_id: int) -> int | None:
        binding = self.get_pepper_binding(pepper_person_id)
        if binding is None:
            return None
        if (
            binding.server_object_id not in self.objects_state
            and binding.server_object_id not in self.tracks
        ):
            return None
        return binding.server_object_id

    def _remove_conflicting_pepper_bindings(
        self,
        *,
        pepper_person_id: int,
        server_object_id: int,
    ):
        stale_pepper_ids = [
            other_pepper_id
            for other_pepper_id, binding in self.pepper_person_bindings.items()
            if other_pepper_id != pepper_person_id
            and binding.server_object_id == server_object_id
        ]
        for other_pepper_id in stale_pepper_ids:
            self.pepper_person_bindings.pop(other_pepper_id, None)

    def upsert_pepper_binding(
        self,
        pepper_person_id: int,
        server_object_id: int,
        *,
        confidence: float = 1.0,
        timestamp: float | None = None,
    ) -> PepperPersonBinding:
        now = timestamp or time.time()
        self._remove_conflicting_pepper_bindings(
            pepper_person_id=pepper_person_id,
            server_object_id=server_object_id,
        )
        binding = self.pepper_person_bindings.get(pepper_person_id)
        if binding is None:
            binding = PepperPersonBinding(
                pepper_person_id=pepper_person_id,
                server_object_id=server_object_id,
                first_seen=now,
                last_seen=now,
                confidence=confidence,
            )
            self.pepper_person_bindings[pepper_person_id] = binding
        else:
            if binding.server_object_id != server_object_id:
                # binding.consecutive_hits += 1
                binding.server_object_id = server_object_id
                # binding.consecutive_hits = 1
                binding.first_seen = min(binding.first_seen, now)
            binding.last_seen = now
            binding.confidence = max(binding.confidence, confidence)
            binding.misses = 0
        self._frame_server_to_pepper[server_object_id] = pepper_person_id
        return binding

    def note_pending_synthetic_pepper_detection(
        self,
        det: InferenceDetectionObject,
        pepper_person_id: int,
    ):
        self._pending_synthetic_pepper_by_detection[id(det)] = pepper_person_id

    def bind_pending_detection_track(
        self,
        det: InferenceDetectionObject,
        *,
        timestamp: float | None = None,
        confidence: float = 1.0,
    ):
        pepper_person_id = self._pending_synthetic_pepper_by_detection.pop(
            id(det), None
        )
        if pepper_person_id is None or det.object_id is None:
            return
        if not isinstance(det.object_id, int):
            return
        self.upsert_pepper_binding(
            pepper_person_id,
            det.object_id,
            confidence=confidence,
            timestamp=timestamp,
        )

    def age_pepper_bindings(
        self,
        seen_pepper_ids: set[int],
        *,
        max_misses: int = 4,
    ):
        stale_pepper_ids: list[int] = []
        for pepper_person_id, binding in self.pepper_person_bindings.items():
            if pepper_person_id in seen_pepper_ids:
                binding.misses = 0
                continue
            binding.misses += 1
            if binding.misses > max_misses:
                stale_pepper_ids.append(pepper_person_id)
        for pepper_person_id in stale_pepper_ids:
            binding = self.pepper_person_bindings.pop(pepper_person_id, None)
            if binding is None:
                continue
            obj = self.objects_state.get(binding.server_object_id)
            if obj is not None and obj.pepper_person_id == pepper_person_id:
                obj.pepper_person_id = None
                obj.robot_distance = None
                obj.robot_engagement_zone = None
                obj.robot_last_seen_ts = None
                obj.attributes = self.merge_person_social_state(obj.attributes, set())

    def upsert_caption(self, caption: SceneCaptionState):
        now = time.time()
        existing = self.captions_state.get(caption.id)
        if existing is None:
            self.captions_state[caption.id] = caption.model_copy(deep=True)
            return
        existing.text = caption.text or existing.text
        existing.provider = caption.provider or existing.provider
        existing.model_id = caption.model_id or existing.model_id
        existing.source = caption.source or existing.source
        existing.frame_id = caption.frame_id or existing.frame_id
        existing.scan_id = caption.scan_id or existing.scan_id
        existing.first_seen = min(existing.first_seen, caption.first_seen or now)
        existing.last_seen = max(existing.last_seen, caption.last_seen or now)
        existing.count = max(existing.count, caption.count or 1)

    def recent_captions(self, limit: int = 5) -> list[SceneCaptionState]:
        if limit <= 0:
            return []
        items = sorted(
            self.captions_state.values(), key=lambda c: c.last_seen, reverse=True
        )
        return items[:limit]
