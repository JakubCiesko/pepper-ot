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
from fastapi import HTTPException
from fastapi import Query

logger = logging.getLogger(__name__)
router = APIRouter()


def _service() -> MemoryService:
    return MemoryService(resolve_runtime_adapter(app_state))


def _normalize_output_language(value: str | None) -> str:
    normalized = str(value or "english").strip().lower()
    if normalized in {"cs", "czech"}:
        return "czech"
    return "english"


def _qa_pool_required():
    if app_state.qa_pool_service is None:
        raise HTTPException(status_code=503, detail="QA pool is not initialized.")
    return app_state.qa_pool_service


async def _force_generate_qa_pool_if_needed(number_of_pairs: int) -> int:
    pool = _qa_pool_required()
    if pool.size() > 0:
        return 0
    if app_state.chat_service is None:
        raise HTTPException(status_code=503, detail="Chat Service is not initialized.")
    memory_service = _service()
    total_memory_description = await run_memory_action(
        lambda: memory_service.build_text_description()
    )
    user_prompt = (
        f"Generate exactly {number_of_pairs} concise question-answer pairs about the current scene.\n"
        "Return only structured data matching the schema.\n"
        "Write every question and every answer in English.\n"
        "Use only facts supported by the provided scene description.\n"
        "Keep each answer short and concrete.\n\n"
        "Scene description:\n"
        f"{total_memory_description}"
    )
    from app.schemas.chat import PregeneratedQAPairs

    structured = await app_state.chat_service.chat_structured(
        user_prompt,
        output_schema=PregeneratedQAPairs,
        conversation_history=None,
        user_prompt_override=user_prompt,
    )
    generated = [
        {"question": item.question.strip(), "answer": item.answer.strip()}
        for item in structured.items
        if item.question.strip() and item.answer.strip()
    ]
    if generated:
        pool.ingest_generated_pairs(generated, source="forced_memory_snapshot")
    return len(generated)


@router.get("/memory", response_model=SceneState)
async def get_memory():
    service = _service()
    logger.info("Memory read requested")
    return await run_memory_action(lambda: service.get_memory())


@router.get("/memory/summary", response_model=MemorySummary)
async def get_memory_summary(
    render_limit: int = Query(5, ge=1, le=20),
    lang: str = Query("en", pattern="^(en|english|cs|czech)$"),
    force_generation: bool = Query(
        False,
        description="If true and QA pool is empty, generate QA from memory snapshot",
    ),
):
    service = _service()
    logger.info(
        "Memory summary requested render_limit=%s lang=%s",
        render_limit,
        lang,
    )
    summary = await run_memory_action(
        lambda: service.get_memory_summary(render_limit=render_limit, lang=lang)
    )
    pool = _qa_pool_required()
    pair_count = (
        app_state.config.qa_generation.pairs_per_update
        if app_state.config is not None
        else 2
    )
    if force_generation:
        await _force_generate_qa_pool_if_needed(max(1, int(pair_count)))
    qa_lang = _normalize_output_language(lang)
    summary.pregenerated_qa = await pool.snapshot_pairs(
        language=qa_lang,
        limit=None,
    )
    return summary


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
    if app_state.qa_pool_service is not None:
        app_state.qa_pool_service.clear()
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
