import time
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from app.core.infra.ws_manager import ws_manager
from app.orchestration.adapters.runtime import memory_payload
from app.orchestration.services.memory_graph_render import MemoryGraphRenderService
from app.schemas.scene import MemorySummary
from app.schemas.scene import Relationship
from app.schemas.scene import SceneState
from app.schemas.scene import TrackedObjectState


class DomainValidationError(ValueError):
    pass


class DomainNotFoundError(KeyError):
    pass


class MemoryObjectCreate(BaseModel):
    id: int | None = None
    label: str
    bbox: list[float] = Field(..., min_length=4, max_length=4)
    status: str = "active"
    source: str = "manual"
    attributes: list[str] = Field(default_factory=list)
    bearing_yaw: float | None = None
    bearing_pitch: float | None = None
    frame_id: str | None = None
    scan_id: str | None = None
    first_seen: float | None = None
    last_seen: float | None = None
    hits: int = Field(1, ge=1)


class MemoryObjectUpdate(BaseModel):
    label: str | None = None
    bbox: list[float] | None = None
    status: str | None = None
    source: str | None = None
    attributes: list[str] | None = None
    bearing_yaw: float | None = None
    bearing_pitch: float | None = None
    frame_id: str | None = None
    scan_id: str | None = None
    first_seen: float | None = None
    last_seen: float | None = None
    hits: int | None = Field(None, ge=1)


class MemoryRelationCreate(BaseModel):
    subject_id: int
    predicate: str
    object_id: int
    first_seen: float | None = None
    last_seen: float | None = None
    count: int = Field(1, ge=1)


class MemoryRelationUpdate(BaseModel):
    subject_id: int
    predicate: str
    object_id: int
    new_subject_id: int | None = None
    new_predicate: str | None = None
    new_object_id: int | None = None
    first_seen: float | None = None
    last_seen: float | None = None
    count: int | None = Field(None, ge=1)


class MemoryService:
    def __init__(self, runtime_adapter):
        self.runtime = runtime_adapter
        self.renderer = MemoryGraphRenderService()

    async def broadcast_current_state(self):
        state = await self.runtime.scene_state()
        await ws_manager.broadcast({"type": "memory", "memory": memory_payload(state)})

    async def get_memory(self) -> SceneState:
        return await self.runtime.scene_state()

    async def get_memory_summary(self, render_limit: int = 5) -> MemorySummary:
        state = await self.runtime.scene_state()
        safe_limit = max(1, min(int(render_limit), self.renderer.MAX_RENDER_OBJECTS))
        render_object_ids = self.renderer.select_render_object_ids(
            state, limit=safe_limit
        )
        crop_map: dict[int, bytes | None] = {}
        getter = getattr(self.runtime, "get_track_crop", None)
        if getter is not None:
            for object_id in render_object_ids:
                crop_map[object_id] = await getter(object_id)
        return self.renderer.build_summary(
            state,
            crop_map=crop_map,
            render_limit=safe_limit,
        )

    async def list_objects(
        self,
        *,
        label: str | None,
        min_hits: int | None,
        skip: int,
        limit: int,
        sort_by: str,
    ) -> dict[str, Any]:
        if sort_by not in {"last_seen", "first_seen", "hits"}:
            raise DomainValidationError(
                "sort_by must be one of last_seen|first_seen|hits"
            )
        state = await self.runtime.scene_state()
        objects = state.objects
        if label:
            objects = [o for o in objects if o.label == label]
        if min_hits is not None:
            objects = [o for o in objects if o.hits >= min_hits]
        objects.sort(key=lambda o: getattr(o, sort_by), reverse=True)
        page = objects[skip : skip + limit]
        return {
            "objects": [o.model_dump(mode="json") for o in page],
            "timestamp": state.timestamp,
        }

    async def list_relations(
        self,
        *,
        subject_id: int | None,
        subject_label: str | None,
        predicate: str | None,
        object_id: int | None,
        object_label: str | None,
        skip: int,
        limit: int,
    ) -> dict[str, Any]:
        state = await self.runtime.scene_state()
        rels = state.relationships
        obj_map = {o.id: o.label for o in state.objects}

        if subject_id is not None:
            rels = [r for r in rels if r.subject_id == subject_id]
        if subject_label is not None:
            rels = [r for r in rels if obj_map.get(r.subject_id) == subject_label]
        if predicate is not None:
            rels = [r for r in rels if r.predicate == predicate]
        if object_id is not None:
            rels = [r for r in rels if r.object_id == object_id]
        if object_label is not None:
            rels = [r for r in rels if obj_map.get(r.object_id) == object_label]

        page = rels[skip : skip + limit]
        return {
            "relationships": [r.model_dump(mode="json") for r in page],
            "timestamp": state.timestamp,
        }

    async def upsert_memory(self, state: SceneState):
        if not state.objects and not state.relationships:
            raise DomainValidationError("SceneState is empty")
        await self.runtime.upsert_scene_state(state)

    async def reset_memory(self, confirm: bool):
        if not confirm:
            raise DomainValidationError(
                "Reset not confirmed. Set confirm=true to proceed."
            )
        await self.runtime.reset_memory()

    async def create_object(self, payload: MemoryObjectCreate) -> TrackedObjectState:
        now = time.time()
        first_seen = payload.first_seen if payload.first_seen is not None else now
        last_seen = payload.last_seen if payload.last_seen is not None else now
        if last_seen < first_seen:
            raise DomainValidationError("last_seen cannot be < first_seen")
        object_id = (
            payload.id
            if payload.id is not None
            else await self.runtime.next_object_id()
        )

        obj = TrackedObjectState(
            id=object_id,
            label=payload.label,
            status=payload.status,
            source=payload.source,
            attributes=payload.attributes,
            bearing_yaw=payload.bearing_yaw,
            bearing_pitch=payload.bearing_pitch,
            frame_id=payload.frame_id,
            scan_id=payload.scan_id,
            first_seen=first_seen,
            last_seen=last_seen,
            hits=payload.hits,
            bbox=payload.bbox,
        )
        await self.runtime.create_object(obj)
        return obj

    async def update_object(
        self, object_id: int, payload: MemoryObjectUpdate
    ) -> TrackedObjectState:
        updates = payload.model_dump(exclude_none=True)
        if not updates:
            raise DomainValidationError("No fields to update")
        if "bbox" in updates and len(updates["bbox"]) != 4:
            raise DomainValidationError("bbox must contain exactly 4 values")
        if (
            "first_seen" in updates
            and "last_seen" in updates
            and updates["last_seen"] < updates["first_seen"]
        ):
            raise DomainValidationError("last_seen cannot be < first_seen")
        try:
            return await self.runtime.patch_object(object_id, updates)
        except KeyError as exc:
            raise DomainNotFoundError(str(exc)) from exc

    async def delete_object(self, object_id: int, cascade_relations: bool):
        deleted = await self.runtime.delete_object(object_id, cascade_relations)
        if not deleted:
            raise DomainNotFoundError(f"Object id={object_id} not found")

    async def create_relation(self, payload: MemoryRelationCreate) -> Relationship:
        now = time.time()
        first_seen = payload.first_seen if payload.first_seen is not None else now
        last_seen = payload.last_seen if payload.last_seen is not None else now
        if last_seen < first_seen:
            raise DomainValidationError("last_seen cannot be < first_seen")

        rel = Relationship(
            subject_id=payload.subject_id,
            predicate=payload.predicate,
            object_id=payload.object_id,
            first_seen=first_seen,
            last_seen=last_seen,
            count=payload.count,
        )
        await self.runtime.create_relation(rel)
        return rel

    async def update_relation(self, payload: MemoryRelationUpdate) -> Relationship:
        updates = {
            "subject_id": payload.new_subject_id,
            "predicate": payload.new_predicate,
            "object_id": payload.new_object_id,
            "first_seen": payload.first_seen,
            "last_seen": payload.last_seen,
            "count": payload.count,
        }
        updates = {k: v for k, v in updates.items() if v is not None}
        if not updates:
            raise DomainValidationError("No fields to update")
        if (
            "first_seen" in updates
            and "last_seen" in updates
            and updates["last_seen"] < updates["first_seen"]
        ):
            raise DomainValidationError("last_seen cannot be < first_seen")

        try:
            return await self.runtime.patch_relation(
                payload.subject_id,
                payload.predicate,
                payload.object_id,
                updates,
            )
        except KeyError as exc:
            raise DomainNotFoundError(str(exc)) from exc

    async def delete_relation(self, subject_id: int, predicate: str, object_id: int):
        deleted = await self.runtime.delete_relation(subject_id, predicate, object_id)
        if not deleted:
            raise DomainNotFoundError(
                f"Relationship ({subject_id}, {predicate}, {object_id}) not found"
            )
