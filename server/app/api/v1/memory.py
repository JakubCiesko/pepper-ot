import logging

from app.core.runtime.state import ml_state
from app.orchestration.memory_service import DomainNotFoundError
from app.orchestration.memory_service import DomainValidationError
from app.orchestration.memory_service import MemoryObjectCreate
from app.orchestration.memory_service import MemoryObjectUpdate
from app.orchestration.memory_service import MemoryRelationCreate
from app.orchestration.memory_service import MemoryRelationUpdate
from app.orchestration.memory_service import MemoryService
from app.orchestration.runtime_adapter import resolve_runtime_adapter
from app.schemas.scene import SceneState
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Query

logger = logging.getLogger(__name__)
router = APIRouter()


def _service() -> MemoryService:
    return MemoryService(resolve_runtime_adapter(ml_state))


@router.get("/memory", response_model=SceneState)
async def get_memory():
    try:
        return await _service().get_memory()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/memory/objects")
async def get_memory_objects(
    label: str | None = None,
    min_hits: int | None = Query(None, ge=1),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1),
    sort_by: str = Query("last_seen", pattern="^(last_seen|first_seen|hits)$"),
):
    try:
        return await _service().list_objects(
            label=label,
            min_hits=min_hits,
            skip=skip,
            limit=limit,
            sort_by=sort_by,
        )
    except DomainValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


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
    try:
        return await _service().list_relations(
            subject_id=subject_id,
            subject_label=subject_label,
            predicate=predicate,
            object_id=object_id,
            object_label=object_label,
            skip=skip,
            limit=limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/memory/upsert")
async def upsert_memory(state: SceneState):
    service = _service()
    try:
        await service.upsert_memory(state)
        await service.broadcast_current_state()
        return {"ok": True}
    except DomainValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/memory/reset")
async def reset_memory(
    confirm: bool = Query(False, description="Must be True to actually reset")
):
    service = _service()
    try:
        await service.reset_memory(confirm)
        await service.broadcast_current_state()
        return {"ok": True}
    except DomainValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/memory/object")
async def create_memory_object(payload: MemoryObjectCreate):
    service = _service()
    try:
        obj = await service.create_object(payload)
        await service.broadcast_current_state()
        return {"ok": True, "object": obj.model_dump(mode="json")}
    except DomainValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.patch("/memory/object/{object_id}")
async def update_memory_object(object_id: int, payload: MemoryObjectUpdate):
    service = _service()
    try:
        updated = await service.update_object(object_id, payload)
        await service.broadcast_current_state()
        return {"ok": True, "object": updated.model_dump(mode="json")}
    except DomainValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DomainNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.delete("/memory/object/{object_id}")
async def delete_memory_object(
    object_id: int,
    cascade_relations: bool = Query(
        True, description="Delete related relationships together with the object"
    ),
):
    service = _service()
    try:
        await service.delete_object(object_id, cascade_relations)
        await service.broadcast_current_state()
        return {"ok": True}
    except DomainNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/memory/relation")
async def create_memory_relation(payload: MemoryRelationCreate):
    service = _service()
    try:
        rel = await service.create_relation(payload)
        await service.broadcast_current_state()
        return {"ok": True, "relationship": rel.model_dump(mode="json")}
    except DomainValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.patch("/memory/relation")
async def update_memory_relation(payload: MemoryRelationUpdate):
    service = _service()
    try:
        rel = await service.update_relation(payload)
        await service.broadcast_current_state()
        return {"ok": True, "relationship": rel.model_dump(mode="json")}
    except DomainValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DomainNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.delete("/memory/relation")
async def delete_memory_relation(
    subject_id: int = Query(...),
    predicate: str = Query(...),
    object_id: int = Query(...),
):
    service = _service()
    try:
        await service.delete_relation(subject_id, predicate, object_id)
        await service.broadcast_current_state()
        return {"ok": True}
    except DomainNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
