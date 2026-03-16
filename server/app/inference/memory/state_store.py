import logging
import time

from app.inference.memory.state_store_geometry import SceneMemoryStoreGeometryMixin
from app.inference.memory.state_store_objects import SceneMemoryStoreObjectsMixin
from app.inference.memory.state_store_relations import SceneMemoryStoreRelationsMixin
from app.inference.memory.state_store_tracks import SceneMemoryStoreTracksMixin
from app.inference.types import TrackedObject
from app.schemas.scene import Relationship
from app.schemas.scene import SceneState
from app.schemas.scene import TrackedObjectState

logger = logging.getLogger(__name__)


class SceneMemoryStore(
    SceneMemoryStoreTracksMixin,
    SceneMemoryStoreGeometryMixin,
    SceneMemoryStoreObjectsMixin,
    SceneMemoryStoreRelationsMixin,
):
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
        logger.info(
            "SceneMemoryStore initialized max_age=%s max_objects=%s max_relations=%s",
            memory_max_age_seconds,
            memory_max_objects,
            memory_max_relations,
        )

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
        logger.info(
            "Upserted scene state objects=%s relationships=%s",
            len(state.objects),
            len(state.relationships),
        )
