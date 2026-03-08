import asyncio
import base64
import io
import time
from typing import Any

import numpy as np
from PIL import Image

from app.core.pipeline_factory import build_visual_pipeline
from app.core.worker_types import WorkerState
from app.schemas.config import AppConfig
from app.schemas.scene import Relationship
from app.schemas.scene import SceneState
from app.schemas.scene import TrackedObjectState


class WorkerRuntime:
    def __init__(self):
        self.config: AppConfig | None = None
        self.pipeline = None
        self.state = WorkerState.STOPPED
        self.started_at = time.time()
        self.last_active = time.time()
        self.inflight_count = 0
        self.last_error: str | None = None
        self.config_version = 0
        self._lock = asyncio.Lock()

    async def apply_config(self, cfg: AppConfig, version: int, rebuild: bool = True):
        async with self._lock:
            self.config = cfg
            self.config_version = version
            if rebuild and self.pipeline is not None:
                self.pipeline = None
            self.state = WorkerState.READY

    async def ensure_pipeline(self):
        if self.config is None:
            raise RuntimeError("worker config is not loaded")
        if self.pipeline is None:
            self.state = WorkerState.STARTING
            self.pipeline = build_visual_pipeline(self.config)
            self.state = WorkerState.READY

    async def warmup(self):
        await self.ensure_pipeline()
        self.last_active = time.time()

    async def detect(self, image_b64: str, robot_metadata) -> dict[str, Any]:
        await self.ensure_pipeline()
        self.state = WorkerState.BUSY
        self.inflight_count += 1
        try:
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            result = await self.pipeline.process(image, robot_metadata)
            som = result.som_image if result.som_image is not None else np.array(image)
            som_pil = Image.fromarray(som.astype("uint8"))
            buf = io.BytesIO()
            som_pil.save(buf, format="JPEG")
            image_out = base64.b64encode(buf.getvalue()).decode("utf-8")
            objects = [
                {
                    "label": det.label,
                    "confidence": det.confidence,
                    "bbox": det.bbox,
                    "object_id": det.object_id,
                }
                for det in result.detections
            ]
            scene_graph = result.scene_graph.as_dict() if result.scene_graph else []
            memory = self.pipeline.memory.scene_state().model_dump()
            self.last_active = time.time()
            return {
                "ok": True,
                "image_b64": image_out,
                "objects": objects,
                "scene_graph": scene_graph,
                "metrics": result.metrics,
                "executed_stages": result.executed_stages,
                "memory": memory,
                "image_width": image.width,
                "image_height": image.height,
                "worker_state": WorkerState.READY,
                "config_version": self.config_version,
            }
        finally:
            self.inflight_count = max(0, self.inflight_count - 1)
            self.state = WorkerState.READY

    async def scene_state(self) -> SceneState:
        await self.ensure_pipeline()
        return self.pipeline.memory.scene_state()

    async def upsert_scene_state(self, state: SceneState):
        await self.ensure_pipeline()
        self.pipeline.memory.upsert_scene_state(state)

    async def reset_memory(self):
        await self.ensure_pipeline()
        self.pipeline.memory.reset()

    async def create_object(self, obj: TrackedObjectState):
        await self.ensure_pipeline()
        self.pipeline.memory.create_object(obj)

    async def patch_object(self, object_id: int, updates: dict):
        await self.ensure_pipeline()
        return self.pipeline.memory.patch_object(object_id, updates)

    async def delete_object(self, object_id: int, cascade_relations: bool):
        await self.ensure_pipeline()
        return self.pipeline.memory.delete_object(
            object_id, cascade_relations=cascade_relations
        )

    async def create_relation(self, rel: Relationship):
        await self.ensure_pipeline()
        self.pipeline.memory.create_relation(rel)

    async def patch_relation(
        self, subject_id: int, predicate: str, object_id: int, updates: dict
    ):
        await self.ensure_pipeline()
        return self.pipeline.memory.patch_relation(
            subject_id, predicate, object_id, updates
        )

    async def delete_relation(self, subject_id: int, predicate: str, object_id: int):
        await self.ensure_pipeline()
        return self.pipeline.memory.delete_relation(subject_id, predicate, object_id)

    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "state": self.state,
            "worker_state": self.state,
            "pid": None,
            "uptime_seconds": max(0.0, time.time() - self.started_at),
            "inflight_count": self.inflight_count,
            "last_active_ts": self.last_active,
            "config_version": self.config_version,
            "restart_count": 0,
            "idle_kill_count": 0,
            "crash_count": 0,
            "breaker_open_until": None,
            "last_error": self.last_error,
        }
