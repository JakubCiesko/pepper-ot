import logging
import os
import signal
import time
from typing import Any

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Query

from app.core.worker_protocol import DetectRPCRequest
from app.core.worker_protocol import WorkerConfigRPCRequest
from app.schemas.config import AppConfig
from app.schemas.scene import Relationship
from app.schemas.scene import SceneState
from app.schemas.scene import TrackedObjectState
from app.worker.runtime import WorkerRuntime

logger = logging.getLogger(__name__)


def build_worker_router(runtime: WorkerRuntime) -> APIRouter:
    router = APIRouter()

    @router.get("/internal/health")
    async def health():
        return {"ok": True, "state": runtime.state}

    @router.get("/internal/status")
    async def status():
        payload = runtime.status()
        payload["pid"] = __import__("os").getpid()
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
            runtime.pipeline.pipeline_controls = cfg.pipeline_controls
            runtime.pipeline.fusion_config = cfg.fusion
            runtime.pipeline.vis_config = cfg.visualization
            runtime.pipeline.scene_graph_service.mode = cfg.scene_graph.mode
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

    @router.post("/internal/shutdown")
    async def shutdown(_request: dict[str, Any]):
        loop = __import__("asyncio").get_running_loop()
        loop.call_later(0.1, lambda: os.kill(os.getpid(), signal.SIGTERM))
        return {"ok": True}

    @router.get("/internal/memory")
    async def get_memory():
        state = await runtime.scene_state()
        return state.model_dump()

    @router.get("/internal/memory/objects")
    async def get_memory_objects(
        label: str | None = None,
        min_hits: int | None = Query(None, ge=1),
        skip: int = Query(0, ge=0),
        limit: int = Query(50, ge=1),
        sort_by: str = Query("last_seen", pattern="^(last_seen|first_seen|hits)$"),
    ):
        state = await runtime.scene_state()
        objects = state.objects
        if label:
            objects = [o for o in objects if o.label == label]
        if min_hits is not None:
            objects = [o for o in objects if o.hits >= min_hits]
        objects.sort(key=lambda o: getattr(o, sort_by), reverse=True)
        objs_page = objects[skip : skip + limit]
        return {
            "objects": [o.model_dump() for o in objs_page],
            "timestamp": state.timestamp,
        }

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
        state = await runtime.scene_state()
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
        return {
            "relationships": [r.model_dump() for r in rels_page],
            "timestamp": state.timestamp,
        }

    @router.post("/internal/memory/upsert")
    async def upsert_memory(state: SceneState):
        if not state.objects and not state.relationships:
            raise HTTPException(status_code=400, detail="SceneState is empty")
        await runtime.upsert_scene_state(state)
        return {"ok": True}

    @router.post("/internal/memory/reset")
    async def reset_memory(confirm: bool = Query(False)):
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Reset not confirmed. Set confirm=true to proceed.",
            )
        await runtime.reset_memory()
        return {"ok": True}

    @router.post("/internal/memory/object")
    async def create_memory_object(payload: dict[str, Any]):
        now = time.time()
        await runtime.ensure_pipeline()
        memory = runtime.pipeline.memory
        object_id = payload.get("id")
        if object_id is None:
            object_id = memory.next_id
        obj = TrackedObjectState(
            id=int(object_id),
            label=str(payload["label"]),
            status=str(payload.get("status", "active")),
            source=str(payload.get("source", "manual")),
            attributes=list(payload.get("attributes", [])),
            bearing_yaw=payload.get("bearing_yaw"),
            bearing_pitch=payload.get("bearing_pitch"),
            frame_id=payload.get("frame_id"),
            scan_id=payload.get("scan_id"),
            first_seen=float(payload.get("first_seen", now)),
            last_seen=float(payload.get("last_seen", now)),
            hits=int(payload.get("hits", 1)),
            bbox=list(payload["bbox"]),
        )
        await runtime.create_object(obj)
        return {"ok": True, "object": obj.model_dump()}

    @router.patch("/internal/memory/object/{object_id}")
    async def update_memory_object(object_id: int, payload: dict[str, Any]):
        try:
            updated = await runtime.patch_object(object_id, payload)
            return {"ok": True, "object": updated.model_dump()}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.delete("/internal/memory/object/{object_id}")
    async def delete_memory_object(
        object_id: int,
        cascade_relations: bool = Query(True),
    ):
        deleted = await runtime.delete_object(object_id, cascade_relations)
        if not deleted:
            raise HTTPException(
                status_code=404, detail=f"Object id={object_id} not found"
            )
        return {"ok": True}

    @router.post("/internal/memory/relation")
    async def create_memory_relation(payload: Relationship):
        await runtime.create_relation(payload)
        return {"ok": True, "relationship": payload.model_dump()}

    @router.patch("/internal/memory/relation")
    async def update_memory_relation(payload: dict[str, Any]):
        try:
            subject_id = int(payload["subject_id"])
            predicate = str(payload["predicate"])
            object_id = int(payload["object_id"])
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail="Missing relationship identity"
            ) from exc
        updates = {
            "subject_id": payload.get("new_subject_id", payload.get("subject_id_new")),
            "predicate": payload.get("new_predicate", payload.get("predicate_new")),
            "object_id": payload.get("new_object_id", payload.get("object_id_new")),
            "first_seen": payload.get("first_seen"),
            "last_seen": payload.get("last_seen"),
            "count": payload.get("count"),
        }
        updates = {k: v for k, v in updates.items() if v is not None}
        try:
            updated = await runtime.patch_relation(
                subject_id, predicate, object_id, updates
            )
            return {"ok": True, "relationship": updated.model_dump()}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.delete("/internal/memory/relation")
    async def delete_memory_relation(
        subject_id: int = Query(...),
        predicate: str = Query(...),
        object_id: int = Query(...),
    ):
        deleted = await runtime.delete_relation(subject_id, predicate, object_id)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Relationship ({subject_id}, {predicate}, {object_id}) not found",
            )
        return {"ok": True}

    return router
