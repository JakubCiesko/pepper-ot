import logging

from app.api.v1.memory_route_utils import run_memory_action
from app.core.runtime.state import app_state
from app.orchestration.adapters.runtime import resolve_runtime_adapter
from app.orchestration.services.memory import MemoryObjectCreate
from app.orchestration.services.memory import MemoryObjectUpdate
from app.orchestration.services.memory import MemoryRelationCreate
from app.orchestration.services.memory import MemoryRelationUpdate
from app.orchestration.services.memory import MemoryService
from app.schemas.scene import MemorySummary
from app.schemas.scene import SceneState
from fastapi import APIRouter
from fastapi import Query

logger = logging.getLogger(__name__)
router = APIRouter()


def _service() -> MemoryService:
    return MemoryService(resolve_runtime_adapter(app_state))


@router.get("/memory", response_model=SceneState)
async def get_memory():
    service = _service()
    logger.info("Memory read requested")
    return await run_memory_action(lambda: service.get_memory())


@router.get("/memory/summary", response_model=MemorySummary)
async def get_memory_summary(
    render_limit: int = Query(5, ge=1, le=20),
    lang: str = Query("en", pattern="^(en|english|cs|czech)$"),
):
    service = _service()
    logger.info(
        "Memory summary requested render_limit=%s lang=%s",
        render_limit,
        lang,
    )
    return await run_memory_action(
        lambda: service.get_memory_summary(render_limit=render_limit, lang=lang)
    )


@router.get("/memory/object/{object_id}/crop")
async def get_memory_object_crop(object_id: int):
    service = _service()
    logger.info("Memory object crop requested id=%s", object_id)
    return await run_memory_action(lambda: service.get_object_crop(object_id))


@router.get("/memory/objects")
async def get_memory_objects(
    label: str | None = None,
    min_hits: int | None = Query(None, ge=1),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1),
    sort_by: str = Query("last_seen", pattern="^(last_seen|first_seen|hits)$"),
):
    service = _service()
    return await run_memory_action(
        lambda: service.list_objects(
            label=label,
            min_hits=min_hits,
            skip=skip,
            limit=limit,
            sort_by=sort_by,
        )
    )


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
    service = _service()
    return await run_memory_action(
        lambda: service.list_relations(
            subject_id=subject_id,
            subject_label=subject_label,
            predicate=predicate,
            object_id=object_id,
            object_label=object_label,
            skip=skip,
            limit=limit,
        )
    )


@router.post("/memory/upsert")
async def upsert_memory(state: SceneState):
    service = _service()
    logger.info(
        "Memory upsert requested objects=%s relationships=%s",
        len(state.objects),
        len(state.relationships),
    )
    await run_memory_action(
        lambda: service.upsert_memory(state),
        on_success=service.broadcast_current_state,
    )
    return {"ok": True}


@router.post("/memory/reset")
async def reset_memory(
    confirm: bool = Query(False, description="Must be True to actually reset")
):
    service = _service()
    logger.info("Memory reset requested confirm=%s", confirm)
    await run_memory_action(
        lambda: service.reset_memory(confirm),
        on_success=service.broadcast_current_state,
    )
    return {"ok": True}


@router.post("/memory/object")
async def create_memory_object(payload: MemoryObjectCreate):
    service = _service()
    logger.info("Memory object create requested label=%s", payload.label)
    obj = await run_memory_action(
        lambda: service.create_object(payload),
        on_success=service.broadcast_current_state,
    )
    return {"ok": True, "object": obj.model_dump(mode="json")}


@router.patch("/memory/object/{object_id}")
async def update_memory_object(object_id: int, payload: MemoryObjectUpdate):
    service = _service()
    logger.info("Memory object update requested id=%s", object_id)
    updated = await run_memory_action(
        lambda: service.update_object(object_id, payload),
        on_success=service.broadcast_current_state,
    )
    return {"ok": True, "object": updated.model_dump(mode="json")}


@router.delete("/memory/object/{object_id}")
async def delete_memory_object(
    object_id: int,
    cascade_relations: bool = Query(
        True, description="Delete related relationships together with the object"
    ),
):
    service = _service()
    logger.info(
        "Memory object delete requested id=%s cascade_relations=%s",
        object_id,
        cascade_relations,
    )
    await run_memory_action(
        lambda: service.delete_object(object_id, cascade_relations),
        on_success=service.broadcast_current_state,
    )
    return {"ok": True}


@router.post("/memory/relation")
async def create_memory_relation(payload: MemoryRelationCreate):
    service = _service()
    logger.info(
        "Memory relation create requested (%s,%s,%s)",
        payload.subject_id,
        payload.predicate,
        payload.object_id,
    )
    rel = await run_memory_action(
        lambda: service.create_relation(payload),
        on_success=service.broadcast_current_state,
    )
    return {"ok": True, "relationship": rel.model_dump(mode="json")}


@router.patch("/memory/relation")
async def update_memory_relation(payload: MemoryRelationUpdate):
    service = _service()
    logger.info(
        "Memory relation update requested (%s,%s,%s)",
        payload.subject_id,
        payload.predicate,
        payload.object_id,
    )
    rel = await run_memory_action(
        lambda: service.update_relation(payload),
        on_success=service.broadcast_current_state,
    )
    return {"ok": True, "relationship": rel.model_dump(mode="json")}


@router.delete("/memory/relation")
async def delete_memory_relation(
    subject_id: int = Query(...),
    predicate: str = Query(...),
    object_id: int = Query(...),
):
    service = _service()
    logger.info(
        "Memory relation delete requested (%s,%s,%s)",
        subject_id,
        predicate,
        object_id,
    )
    await run_memory_action(
        lambda: service.delete_relation(subject_id, predicate, object_id),
        on_success=service.broadcast_current_state,
    )
    return {"ok": True}
