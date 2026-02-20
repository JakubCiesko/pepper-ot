import logging

from app.core.state import ml_state
from app.schemas.scene import SceneState
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Query

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_memory():
    if ml_state.pipeline is None or ml_state.pipeline.memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    return ml_state.pipeline.memory


@router.get("/memory", response_model=SceneState)
async def get_memory():
    """Return the full current SceneState."""
    memory = _get_memory()
    return memory.scene_state()


@router.get("/memory/objects")
async def get_memory_objects(
    label: str | None = None,
    min_hits: int | None = Query(None, ge=1),
):
    """Return tracked objects, optionally filtered."""
    memory = _get_memory()
    state = memory.scene_state()
    objects = state.objects
    if label:
        objects = [o for o in objects if o.label == label]
    if min_hits is not None:
        objects = [o for o in objects if o.hits >= min_hits]
    return {"objects": [o.model_dump() for o in objects], "timestamp": state.timestamp}


@router.get("/memory/relations")
async def get_memory_relations(
    subject_id: int | None = None,
    subject_label: str | None = None,
    predicate: str | None = None,
    object_id: int | None = None,
    object_label: str | None = None,
):
    """Return relationships, optionally filtered."""
    memory = _get_memory()
    state = memory.scene_state()
    rels = state.relationships
    id_to_label = {o.id: o.label for o in state.objects}
    if subject_id is not None:
        rels = [r for r in rels if r.subject_id == subject_id]
    if subject_label is not None:
        rels = [r for r in rels if id_to_label.get(r.subject_id) == subject_label]
    if predicate is not None:
        rels = [r for r in rels if r.predicate == predicate]
    if object_id is not None:
        rels = [r for r in rels if r.object_id == object_id]
    if object_label is not None:
        rels = [r for r in rels if id_to_label.get(r.object_id) == object_label]
    return {
        "relationships": [r.model_dump() for r in rels],
        "timestamp": state.timestamp,
    }


@router.post("/memory/upsert")
async def upsert_memory(state: SceneState):
    """Merge an external SceneState into memory (manual injection)."""
    memory = _get_memory()
    memory.upsert_scene_state(state)
    logger.info("Upserted SceneState into memory")
    return {"ok": True}


@router.post("/memory/reset")
async def reset_memory():
    """Clear memory (objects + relations)."""
    memory = _get_memory()
    memory.objects_state.clear()
    memory.relations_state.clear()
    memory.tracks.clear()
    memory.next_id = 1
    return {"ok": True}
