import base64
import logging
import os
import signal
from typing import Any

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Query

from app.api.v1.memory_route_utils import run_memory_action
from app.core.config.mutations.runtime import apply_pipeline_runtime_updates
from app.core.config.mutations.runtime import apply_scene_graph_runtime_updates
from app.core.config.mutations.runtime import resolve_base_dir
from app.core.runtime.worker_client.rpc import DetectRPCRequest
from app.core.runtime.worker_client.rpc import WorkerConfigRPCRequest
from app.orchestration.adapters.runtime import WorkerInternalRuntimeAdapter
from app.orchestration.services.memory import MemoryObjectCreate
from app.orchestration.services.memory import MemoryObjectUpdate
from app.orchestration.services.memory import MemoryRelationCreate
from app.orchestration.services.memory import MemoryRelationUpdate
from app.orchestration.services.memory import MemoryService
from app.schemas.config import AppConfig
from app.schemas.scene import MemorySummary
from app.schemas.scene import SceneState
from app.worker.runtime import WorkerRuntime

logger = logging.getLogger(__name__)


def build_worker_router(runtime: WorkerRuntime) -> APIRouter:
    router = APIRouter()

    def memory_service() -> MemoryService:
        return MemoryService(WorkerInternalRuntimeAdapter(runtime))

    @router.get("/internal/health")
    async def health():
        return {"ok": True, "state": runtime.state}

    @router.get("/internal/status")
    async def status():
        payload = runtime.status()
        payload["pid"] = os.getpid()
        return payload

    @router.post("/internal/config/reload")
    async def config_reload(request: WorkerConfigRPCRequest):
        logger.info(
            "Worker internal config reload requested version=%s", request.config_version
        )
        cfg = AppConfig(**request.config)
        await runtime.apply_config(cfg, request.config_version, rebuild=True)
        return {
            "ok": True,
            "worker_state": runtime.state,
            "config_version": runtime.config_version,
        }

    @router.post("/internal/config/hot_update")
    async def config_hot_update(request: dict[str, Any]):
        cfg_raw = request.get("config")
        if not isinstance(cfg_raw, dict):
            raise HTTPException(status_code=400, detail="config must be an object")
        version = int(request.get("config_version", runtime.config_version))
        logger.info("Worker internal hot config update requested version=%s", version)
        cfg = AppConfig(**cfg_raw)
        await runtime.apply_config(cfg, version, rebuild=False)
        if runtime.pipeline is not None:
            base_dir = resolve_base_dir(cfg)
            apply_pipeline_runtime_updates(runtime.pipeline, cfg, base_dir)
            apply_scene_graph_runtime_updates(
                runtime.pipeline.scene_graph_service,
                cfg,
                base_dir,
            )
        await runtime.update_caption_runtime(cfg)
        return {
            "ok": True,
            "worker_state": runtime.state,
            "config_version": runtime.config_version,
        }

    @router.post("/internal/warmup")
    async def warmup(_request: dict[str, Any]):
        logger.info("Worker internal warmup requested")
        await runtime.warmup()
        return {
            "ok": True,
            "worker_state": runtime.state,
            "config_version": runtime.config_version,
        }

    @router.post("/internal/detect")
    async def detect(request: DetectRPCRequest):
        logger.info("Worker internal detect requested")
        return await runtime.detect(request.image_b64, request.robot_metadata)

    @router.post("/internal/caption")
    async def caption(request: dict[str, Any]):
        image_b64 = request.get("image_b64")
        if not isinstance(image_b64, str) or not image_b64:
            raise HTTPException(status_code=400, detail="image_b64 is required")
        prompt = request.get("prompt")
        if prompt is not None and not isinstance(prompt, str):
            raise HTTPException(status_code=400, detail="prompt must be string")
        logger.info("Worker internal caption requested")
        return await runtime.caption(image_b64, prompt_override=prompt)

    @router.post("/internal/vision_chat")
    async def vision_chat(request: dict[str, Any]):
        image_b64 = request.get("image_b64")
        user_prompt = request.get("user_prompt")
        system_prompt = request.get("system_prompt")
        if not isinstance(image_b64, str) or not image_b64:
            raise HTTPException(status_code=400, detail="image_b64 is required")
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise HTTPException(status_code=400, detail="user_prompt is required")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise HTTPException(status_code=400, detail="system_prompt must be string")
        logger.info("Worker internal vision_chat requested")
        return await runtime.vision_chat(
            image_b64,
            user_prompt=user_prompt.strip(),
            system_prompt=(
                system_prompt.strip() if isinstance(system_prompt, str) else None
            ),
        )

    @router.post("/internal/shutdown")
    async def shutdown(_request: dict[str, Any]):
        loop = __import__("asyncio").get_running_loop()
        loop.call_later(0.1, lambda: os.kill(os.getpid(), signal.SIGTERM))
        return {"ok": True}

    @router.get("/internal/memory")
    async def get_memory():
        state = await memory_service().get_memory()
        return state.model_dump(mode="json")

    @router.get("/internal/memory/summary", response_model=MemorySummary)
    async def get_memory_summary(render_limit: int = Query(5, ge=1, le=6)):
        summary = await memory_service().get_memory_summary(render_limit=render_limit)
        return summary.model_dump(mode="json")

    @router.get("/internal/memory/object/{object_id}/crop")
    async def get_memory_object_crop(object_id: int):
        crop_bytes = await runtime.get_track_crop(object_id)
        return {
            "ok": True,
            "object_id": object_id,
            "image_b64": (
                base64.b64encode(crop_bytes).decode("utf-8")
                if crop_bytes is not None
                else None
            ),
        }

    @router.get("/internal/memory/objects")
    async def get_memory_objects(
        label: str | None = None,
        min_hits: int | None = Query(None, ge=1),
        skip: int = Query(0, ge=0),
        limit: int = Query(50, ge=1),
        sort_by: str = Query("last_seen", pattern="^(last_seen|first_seen|hits)$"),
    ):
        service = memory_service()
        return await run_memory_action(
            lambda: service.list_objects(
                label=label,
                min_hits=min_hits,
                skip=skip,
                limit=limit,
                sort_by=sort_by,
            )
        )

    @router.get("/internal/memory/relations")
    async def get_memory_relations(
        subject_id: int | None = None,
        subject_label: str | None = None,
        predicate: str | None = None,
        object_id: int | None = None,
        object_label: str | None = None,
        skip: int = Query(0, ge=0),
        limit: int = Query(50, ge=1),
    ):
        service = memory_service()
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

    @router.post("/internal/memory/upsert")
    async def upsert_memory(state: SceneState):
        service = memory_service()
        await run_memory_action(lambda: service.upsert_memory(state))
        return {"ok": True}

    @router.post("/internal/memory/reset")
    async def reset_memory(confirm: bool = Query(False)):
        service = memory_service()
        await run_memory_action(lambda: service.reset_memory(confirm))
        return {"ok": True}

    @router.post("/internal/memory/object")
    async def create_memory_object(payload: MemoryObjectCreate):
        service = memory_service()
        obj = await run_memory_action(lambda: service.create_object(payload))
        return {"ok": True, "object": obj.model_dump(mode="json")}

    @router.patch("/internal/memory/object/{object_id}")
    async def update_memory_object(object_id: int, payload: MemoryObjectUpdate):
        service = memory_service()
        updated = await run_memory_action(
            lambda: service.update_object(object_id, payload)
        )
        return {"ok": True, "object": updated.model_dump(mode="json")}

    @router.delete("/internal/memory/object/{object_id}")
    async def delete_memory_object(
        object_id: int,
        cascade_relations: bool = Query(True),
    ):
        service = memory_service()
        await run_memory_action(
            lambda: service.delete_object(object_id, cascade_relations)
        )
        return {"ok": True}

    @router.post("/internal/memory/relation")
    async def create_memory_relation(payload: MemoryRelationCreate):
        service = memory_service()
        rel = await run_memory_action(lambda: service.create_relation(payload))
        return {"ok": True, "relationship": rel.model_dump(mode="json")}

    @router.patch("/internal/memory/relation")
    async def update_memory_relation(payload: MemoryRelationUpdate):
        service = memory_service()
        updated = await run_memory_action(lambda: service.update_relation(payload))
        return {"ok": True, "relationship": updated.model_dump(mode="json")}

    @router.delete("/internal/memory/relation")
    async def delete_memory_relation(
        subject_id: int = Query(...),
        predicate: str = Query(...),
        object_id: int = Query(...),
    ):
        service = memory_service()
        await run_memory_action(
            lambda: service.delete_relation(subject_id, predicate, object_id)
        )
        return {"ok": True}

    return router
