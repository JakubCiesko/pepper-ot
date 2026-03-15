import logging

from app.core.runtime.state import app_state
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
    return MemoryService(resolve_runtime_adapter(app_state))


@router.get("/memory", response_model=SceneState)
async def get_memory():
    """
    Retrieve the full current scene memory state.

    Returns:
      SceneState: Current memory snapshot including objects, relationships,
      and timestamp.

    Raises:
      HTTPException: 503 when memory retrieval fails due to runtime/service issues.
    """

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
    """
    List memory objects with filtering, pagination, and sorting.

    Args:
      label: Optional exact/normalized label filter.
      min_hits: Optional minimum observation-count filter.
      skip: Pagination offset.
      limit: Maximum number of returned records.
      sort_by: Sort key (`last_seen`, `first_seen`, or `hits`).

    Returns:
      dict: Object listing payload from MemoryService.

    Raises:
      HTTPException:
          - 400 for invalid domain/query constraints.
          - 503 for runtime/service failures.
    """
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
    """
    List memory relationships with optional subject/object/predicate filters.

    Args:
      subject_id: Filter by subject object ID.
      subject_label: Filter by subject label.
      predicate: Filter by relation predicate.
      object_id: Filter by object ID.
      object_label: Filter by object label.
      skip: Pagination offset.
      limit: Maximum number of returned records.

    Returns:
      dict: Relationship listing payload from MemoryService.

    Raises:
      HTTPException: 503 for runtime/service failures.
    """

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
    """
    Replace memory state with a provided scene snapshot.

    After successful upsert, broadcasts the updated memory state to subscribers.

    Args:
        state: Full SceneState to write into memory.

    Returns:
        dict: `{"ok": True}` on success.

    Raises:
        HTTPException:
            - 400 for domain validation failures.
            - 503 for runtime/service failures.
    """

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
    """
    Reset scene memory to an empty state.

    Safety behavior:
    - Requires `confirm=true` to prevent accidental resets.

    After successful reset, broadcasts the updated (empty) memory state.

    Args:
        confirm: Must be True to execute reset.

    Returns:
        dict: `{"ok": True}` on success.

    Raises:
        HTTPException:
            - 400 when confirm is missing/false or domain validation fails.
            - 503 for runtime/service failures.
    """

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
    """
    Create a new tracked object in memory.

    After successful creation, broadcasts updated memory state.

    Args:
        payload: MemoryObjectCreate object payload.

    Returns:
        dict:
            {
              "ok": True,
              "object": <serialized object>
            }

    Raises:
        HTTPException:
            - 400 for invalid object payload/domain validation.
            - 503 for runtime/service failures.
    """

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
    """
    Update an existing memory object by object ID.

    After successful update, broadcasts updated memory state.

    Args:
        object_id: Target object identifier.
        payload: Partial update payload.

    Returns:
        dict:
            {
              "ok": True,
              "object": <serialized updated object>
            }

    Raises:
        HTTPException:
            - 400 for invalid update payload/domain validation.
            - 404 when object is not found.
            - 503 for runtime/service failures.
    """

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
    """
    Delete a memory object and optionally cascade-delete linked relationships.

    After successful deletion, broadcasts updated memory state.

    Args:
        object_id: Target object identifier.
        cascade_relations: If True, remove relationships referencing this object.

    Returns:
        dict: `{"ok": True}` on success.

    Raises:
        HTTPException:
            - 404 when object is not found.
            - 503 for runtime/service failures.
    """

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
    """
    Create a new relationship in scene memory.

    After successful creation, broadcasts updated memory state.

    Args:
        payload: MemoryRelationCreate payload.

    Returns:
        dict:
            {
              "ok": True,
              "relationship": <serialized relationship>
            }

    Raises:
        HTTPException:
            - 400 for invalid relation payload/domain validation.
            - 503 for runtime/service failures.
    """

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
    """
    Update an existing relationship in scene memory.

    After successful update, broadcasts updated memory state.

    Args:
        payload: MemoryRelationUpdate payload identifying and updating a relation.

    Returns:
        dict:
            {
              "ok": True,
              "relationship": <serialized updated relationship>
            }

    Raises:
        HTTPException:
            - 400 for invalid relation payload/domain validation.
            - 404 when relationship is not found.
            - 503 for runtime/service failures.
    """

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
    """
    Delete a relationship identified by (subject_id, predicate, object_id).

    After successful deletion, broadcasts updated memory state.

    Args:
        subject_id: Subject object ID.
        predicate: Relationship predicate.
        object_id: Object ID.

    Returns:
        dict: `{"ok": True}` on success.

    Raises:
        HTTPException:
            - 404 when relationship is not found.
            - 503 for runtime/service failures.
    """

    service = _service()
    try:
        await service.delete_relation(subject_id, predicate, object_id)
        await service.broadcast_current_state()
        return {"ok": True}
    except DomainNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
