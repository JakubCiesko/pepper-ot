import logging
import os
import signal
from typing import Any

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Query

from app.core.config.runtime_mutations import apply_pipeline_runtime_updates
from app.core.config.runtime_mutations import apply_scene_graph_runtime_updates
from app.core.config.runtime_mutations import resolve_base_dir
from app.core.runtime.worker_client.rpc import DetectRPCRequest
from app.core.runtime.worker_client.rpc import WorkerConfigRPCRequest
from app.orchestration.memory_service import DomainNotFoundError
from app.orchestration.memory_service import DomainValidationError
from app.orchestration.memory_service import MemoryObjectCreate
from app.orchestration.memory_service import MemoryObjectUpdate
from app.orchestration.memory_service import MemoryRelationCreate
from app.orchestration.memory_service import MemoryRelationUpdate
from app.orchestration.memory_service import MemoryService
from app.orchestration.runtime_adapter import WorkerProcessRuntimeAdapter
from app.schemas.config import AppConfig
from app.schemas.scene import SceneState
from app.worker.runtime import WorkerRuntime

logger = logging.getLogger(__name__)


def build_worker_router(runtime: WorkerRuntime) -> APIRouter:
    router = APIRouter()

    def memory_service() -> MemoryService:
        return MemoryService(WorkerProcessRuntimeAdapter(runtime))

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
        await runtime.warmup()
        return {
            "ok": True,
            "worker_state": runtime.state,
            "config_version": runtime.config_version,
        }

    @router.post("/internal/detect")
    async def detect(request: DetectRPCRequest):
        return await runtime.detect(request.image_b64, request.robot_metadata)

    @router.post("/internal/caption")
    async def caption(request: dict[str, Any]):
        image_b64 = request.get("image_b64")
        if not isinstance(image_b64, str) or not image_b64:
            raise HTTPException(status_code=400, detail="image_b64 is required")
        prompt = request.get("prompt")
        if prompt is not None and not isinstance(prompt, str):
            raise HTTPException(status_code=400, detail="prompt must be string")
        return await runtime.caption(image_b64, prompt_override=prompt)

    @router.post("/internal/shutdown")
    async def shutdown(_request: dict[str, Any]):
        loop = __import__("asyncio").get_running_loop()
        loop.call_later(0.1, lambda: os.kill(os.getpid(), signal.SIGTERM))
        return {"ok": True}

    @router.get("/internal/memory")
    async def get_memory():
        state = await memory_service().get_memory()
        return state.model_dump(mode="json")

    @router.get("/internal/memory/objects")
    async def get_memory_objects(
        label: str | None = None,
        min_hits: int | None = Query(None, ge=1),
        skip: int = Query(0, ge=0),
        limit: int = Query(50, ge=1),
        sort_by: str = Query("last_seen", pattern="^(last_seen|first_seen|hits)$"),
    ):
        try:
            return await memory_service().list_objects(
                label=label,
                min_hits=min_hits,
                skip=skip,
                limit=limit,
                sort_by=sort_by,
            )
        except DomainValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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
        return await memory_service().list_relations(
            subject_id=subject_id,
            subject_label=subject_label,
            predicate=predicate,
            object_id=object_id,
            object_label=object_label,
            skip=skip,
            limit=limit,
        )

    @router.post("/internal/memory/upsert")
    async def upsert_memory(state: SceneState):
        try:
            await memory_service().upsert_memory(state)
            return {"ok": True}
        except DomainValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/internal/memory/reset")
    async def reset_memory(confirm: bool = Query(False)):
        try:
            await memory_service().reset_memory(confirm)
            return {"ok": True}
        except DomainValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/internal/memory/object")
    async def create_memory_object(payload: MemoryObjectCreate):
        try:
            obj = await memory_service().create_object(payload)
            return {"ok": True, "object": obj.model_dump(mode="json")}
        except DomainValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.patch("/internal/memory/object/{object_id}")
    async def update_memory_object(object_id: int, payload: MemoryObjectUpdate):
        try:
            updated = await memory_service().update_object(object_id, payload)
            return {"ok": True, "object": updated.model_dump(mode="json")}
        except DomainValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except DomainNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.delete("/internal/memory/object/{object_id}")
    async def delete_memory_object(
        object_id: int,
        cascade_relations: bool = Query(True),
    ):
        try:
            await memory_service().delete_object(object_id, cascade_relations)
            return {"ok": True}
        except DomainNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/internal/memory/relation")
    async def create_memory_relation(payload: MemoryRelationCreate):
        try:
            rel = await memory_service().create_relation(payload)
            return {"ok": True, "relationship": rel.model_dump(mode="json")}
        except DomainValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.patch("/internal/memory/relation")
    async def update_memory_relation(payload: MemoryRelationUpdate):
        try:
            updated = await memory_service().update_relation(payload)
            return {"ok": True, "relationship": updated.model_dump(mode="json")}
        except DomainValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except DomainNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.delete("/internal/memory/relation")
    async def delete_memory_relation(
        subject_id: int = Query(...),
        predicate: str = Query(...),
        object_id: int = Query(...),
    ):
        try:
            await memory_service().delete_relation(subject_id, predicate, object_id)
            return {"ok": True}
        except DomainNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return router
