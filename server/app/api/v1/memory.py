import logging
import time

from app.core.state import ml_state
from app.core.ws_manager import ws_manager
from app.schemas.scene import SceneState
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Query

logger = logging.getLogger(__name__)
router = APIRouter()


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
        "timestamp": memory.scene_state().timestamp,
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
    except Exception as e:
        logger.error(f"Failed to upsert memory: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to upsert memory: {e}"
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
    memory.objects_state.clear()
    memory.relations_state.clear()
    memory.tracks.clear()
    memory.next_id = 1
    logger.info("Memory reset successfully")
    current = memory.scene_state()
    await broadcast_memory(current)
    return {"ok": True}
