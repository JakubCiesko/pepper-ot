import logging
import time

from app.core.state import ml_state
from app.core.ws_manager import ws_manager
from app.schemas.scene import Relationship
from app.schemas.scene import SceneState
from app.schemas.scene import TrackedObjectState
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Query
from pydantic import BaseModel
from pydantic import Field

logger = logging.getLogger(__name__)
router = APIRouter()


class MemoryObjectCreateRequest(BaseModel):
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


class MemoryObjectUpdateRequest(BaseModel):
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


class MemoryRelationCreateRequest(BaseModel):
    subject_id: int
    predicate: str
    object_id: int
    first_seen: float | None = None
    last_seen: float | None = None
    count: int = Field(1, ge=1)


class MemoryRelationUpdateRequest(BaseModel):
    subject_id: int
    predicate: str
    object_id: int
    new_subject_id: int | None = None
    new_predicate: str | None = None
    new_object_id: int | None = None
    first_seen: float | None = None
    last_seen: float | None = None
    count: int | None = Field(None, ge=1)


def _get_memory():
    if ml_state.pipeline is None or ml_state.pipeline.memory is None:
        logger.warning("Memory requested but not initialized")
        raise HTTPException(status_code=503, detail="Memory not initialized")
    return ml_state.pipeline.memory


async def broadcast_memory(state: SceneState):
    """broadcast memory to dashboard"""
    payload = {
        "type": "memory",
        "memory": (
            state.model_dump()
            if state
            else {"objects": [], "relationships": [], "timestamp": time.time()}
        ),
    }
    await ws_manager.broadcast(payload)


@router.get("/memory", response_model=SceneState)
async def get_memory():
    """
    Return the full current SceneState, which is a representation of all objects currently stored in memory (past and present).
    Returns:
        SceneState: All tracked objects and relationships.
    """
    memory = _get_memory()
    return memory.scene_state()


@router.get("/memory/objects")
async def get_memory_objects(
    label: str | None = None,
    min_hits: int | None = Query(None, ge=1),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1),
    sort_by: str = Query("last_seen", regex="^(last_seen|first_seen|hits)$"),
):
    """
    Return tracked objects, optionally filtered and paginated.

    Args:
        label: Filter objects by label.
        min_hits: Only include objects with hits >= min_hits.
        skip: Number of objects to skip (pagination).
        limit: Max number of objects to return.
        sort_by: Sort objects by 'last_seen', 'first_seen', or 'hits'.

    Returns:
        dict: Contains list of serialized objects and memory timestamp.
    """
    memory = _get_memory()
    state = memory.scene_state()
    objects = state.objects
    if label:
        objects = [o for o in objects if o.label == label]
    if min_hits is not None:
        objects = [o for o in objects if o.hits >= min_hits]
    objects.sort(key=lambda o: getattr(o, sort_by), reverse=True)
    objs_page = objects[skip : skip + limit]
    logger.info(
        f"Returning {len(objs_page)} objects (total={len(state.objects)}, "
        f"filter: label={label}, min_hits={min_hits}, sort_by={sort_by})"
    )
    return {
        "objects": [o.model_dump() for o in objs_page],
        "timestamp": state.timestamp,
    }


@router.get("/memory/relations")
async def get_memory_relations(
    subject_id: int | None = None,
    subject_label: str | None = None,
    predicate: str | None = None,
    object_id: int | None = None,
    object_label: str | None = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1),
):
    """
    Return relationships from memory, optionally filtered and paginated.

    Args:
        subject_id: Filter by subject object ID.
        subject_label: Filter by subject object label.
        predicate: Filter by predicate string.
        object_id: Filter by object ID.
        object_label: Filter by object label.
        skip: Number of relationships to skip (pagination).
        limit: Max number of relationships to return.

    Returns:
        dict: Contains list of serialized relationships and memory timestamp.
    """
    memory = _get_memory()
    state = memory.scene_state()
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

    rels_page = rels[skip : skip + limit]
    logger.info(
        f"Returning {len(rels_page)} relationships (total={len(rels)}, "
        f"filters applied)"
    )
    return {
        "relationships": [r.model_dump() for r in rels_page],
        "timestamp": state.timestamp,
    }


@router.post("/memory/upsert")
async def upsert_memory(state: SceneState):
    """
    Merge an external SceneState into memory (manual injection).

    Args:
        state: SceneState object to upsert into memory.

    Returns:
        dict: Status of operation.
    """
    memory = _get_memory()
    if not state.objects and not state.relationships:
        raise HTTPException(status_code=400, detail="SceneState is empty")
    try:
        memory.upsert_scene_state(state)
        logger.info("Upserted SceneState into memory successfully")
        current = memory.scene_state()
        await broadcast_memory(current)
        return {"ok": True}
    except ValueError as e:
        logger.exception(
            f"Inappropriate argument value when upserting state to memory: {e}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception(f"Failed to upsert memory: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to upsert memory: {e}"
        ) from e


@router.post("/memory/reset")
async def reset_memory(
    confirm: bool = Query(False, description="Must be True to actually reset")
):
    """
    Clear memory (objects + relationships). Requires explicit confirmation. Needs safety double check of setting confirm true.

    Args:
        confirm: Must be True to perform reset.

    Returns:
        dict: Status of operation.
    """
    if not confirm:
        raise HTTPException(
            status_code=400, detail="Reset not confirmed. Set confirm=true to proceed."
        )
    memory = _get_memory()
    # memory.objects_state.clear()
    # memory.relations_state.clear()
    # memory.tracks.clear()
    # memory.next_id = 1
    memory.reset()
    logger.info("Memory reset successfully")
    current = memory.scene_state()
    await broadcast_memory(current)
    return {"ok": True}


@router.post("/memory/object")
async def create_memory_object(payload: MemoryObjectCreateRequest):
    memory = _get_memory()
    now = time.time()
    object_id = payload.id if payload.id is not None else memory.next_id
    first_seen = payload.first_seen if payload.first_seen is not None else now
    last_seen = payload.last_seen if payload.last_seen is not None else now
    if last_seen < first_seen:
        raise HTTPException(status_code=400, detail="last_seen cannot be < first_seen")
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
    try:
        memory.create_object(obj)
        current = memory.scene_state()
        await broadcast_memory(current)
        return {"ok": True, "object": obj.model_dump()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.patch("/memory/object/{object_id}")
async def update_memory_object(object_id: int, payload: MemoryObjectUpdateRequest):
    memory = _get_memory()
    updates = payload.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    if "bbox" in updates and len(updates["bbox"]) != 4:
        raise HTTPException(
            status_code=400, detail="bbox must contain exactly 4 values"
        )
    if (
        "first_seen" in updates
        and "last_seen" in updates
        and updates["last_seen"] < updates["first_seen"]
    ):
        raise HTTPException(status_code=400, detail="last_seen cannot be < first_seen")
    try:
        updated = memory.patch_object(object_id, updates)
        current = memory.scene_state()
        await broadcast_memory(current)
        return {"ok": True, "object": updated.model_dump()}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.delete("/memory/object/{object_id}")
async def delete_memory_object(
    object_id: int,
    cascade_relations: bool = Query(
        True, description="Delete related relationships together with the object"
    ),
):
    memory = _get_memory()
    deleted = memory.delete_object(object_id, cascade_relations=cascade_relations)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Object id={object_id} not found")
    current = memory.scene_state()
    await broadcast_memory(current)
    return {"ok": True}


@router.post("/memory/relation")
async def create_memory_relation(payload: MemoryRelationCreateRequest):
    memory = _get_memory()
    now = time.time()
    first_seen = payload.first_seen if payload.first_seen is not None else now
    last_seen = payload.last_seen if payload.last_seen is not None else now
    if last_seen < first_seen:
        raise HTTPException(status_code=400, detail="last_seen cannot be < first_seen")
    rel = Relationship(
        subject_id=payload.subject_id,
        predicate=payload.predicate,
        object_id=payload.object_id,
        first_seen=first_seen,
        last_seen=last_seen,
        count=payload.count,
    )
    try:
        memory.create_relation(rel)
        current = memory.scene_state()
        await broadcast_memory(current)
        return {"ok": True, "relationship": rel.model_dump()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.patch("/memory/relation")
async def update_memory_relation(payload: MemoryRelationUpdateRequest):
    memory = _get_memory()
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
        raise HTTPException(status_code=400, detail="No fields to update")
    if (
        "first_seen" in updates
        and "last_seen" in updates
        and updates["last_seen"] < updates["first_seen"]
    ):
        raise HTTPException(status_code=400, detail="last_seen cannot be < first_seen")
    try:
        updated = memory.patch_relation(
            payload.subject_id, payload.predicate, payload.object_id, updates
        )
        current = memory.scene_state()
        await broadcast_memory(current)
        return {"ok": True, "relationship": updated.model_dump()}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.delete("/memory/relation")
async def delete_memory_relation(
    subject_id: int = Query(...),
    predicate: str = Query(...),
    object_id: int = Query(...),
):
    memory = _get_memory()
    deleted = memory.delete_relation(subject_id, predicate, object_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Relationship ({subject_id}, {predicate}, {object_id}) not found",
        )
    current = memory.scene_state()
    await broadcast_memory(current)
    return {"ok": True}
