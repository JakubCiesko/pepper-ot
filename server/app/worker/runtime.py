import asyncio
import base64
import io
import time
from typing import Any

import numpy as np
from PIL import Image

from app.core.pipeline_factory import build_visual_pipeline
from app.core.runtime.worker_types import WorkerState
from app.providers.caption_client import CaptionClient
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
        self.caption_client: CaptionClient | None = None
        self.caption_system_prompt: str = ""
        self.caption_user_prompt: str | None = None

    async def apply_config(self, cfg: AppConfig, version: int, rebuild: bool = True):
        async with self._lock:
            self.config = cfg
            self.config_version = version
            if rebuild and self.pipeline is not None:
                self.pipeline = None
            if rebuild and self.caption_client is not None:
                self.caption_client = None
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

    def _resolve_caption_prompt(self, source, default: str | None = None) -> str | None:
        if source is None:
            return default
        base_dir = (
            self.config._config_path.parent
            if self.config is not None and self.config._config_path is not None
            else None
        )
        if source.text is not None:
            return source.text.strip()
        if source.path is not None and base_dir is not None:
            return source.resolve(base_dir)
        return default

    async def ensure_caption_client(self):
        if self.config is None:
            raise RuntimeError("worker config is not loaded")
        if self.caption_client is None:
            self.caption_client = CaptionClient(self.config.caption)
            self.caption_system_prompt = (
                self._resolve_caption_prompt(self.config.caption.system_prompt, "")
                or ""
            )
            self.caption_user_prompt = self._resolve_caption_prompt(
                self.config.caption.user_prompt, None
            )

    async def update_caption_runtime(self, cfg: AppConfig):
        if self.caption_client is None:
            return
        self.caption_system_prompt = (
            self._resolve_caption_prompt(
                cfg.caption.system_prompt, self.caption_system_prompt
            )
            or ""
        )
        self.caption_user_prompt = self._resolve_caption_prompt(
            cfg.caption.user_prompt, self.caption_user_prompt
        )
        self.caption_client.update_runtime(cfg.caption, rebuild_client=False)

    def _final_caption_prompt(self, prompt_override: str | None) -> str:
        if prompt_override and prompt_override.strip():
            prompt = prompt_override.strip()
        elif self.config is not None and self.config.caption.mode == "unconditional":
            prompt = ""
        else:
            prompt = (self.caption_user_prompt or "").strip()

        if (
            self.config is not None
            and self.config.caption.max_words is not None
            and self.config.caption.max_words > 0
        ):
            prompt = (
                f"{prompt}\nRespond in at most {int(self.config.caption.max_words)} words."
            ).strip()
        return prompt

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

    async def caption(
        self, image_b64: str, prompt_override: str | None = None
    ) -> dict[str, Any]:
        await self.ensure_caption_client()
        assert self.caption_client is not None
        image_bytes = base64.b64decode(image_b64)
        prompt = self._final_caption_prompt(prompt_override)
        text = await self.caption_client.infer(
            self.caption_system_prompt,
            prompt,
            image_bytes,
        )
        self.last_active = time.time()
        return {
            "ok": True,
            "caption": text,
            "provider": self.config.caption.provider if self.config else "",
            "model_id": self.config.caption.model_id if self.config else "",
            "worker_state": WorkerState.READY,
            "config_version": self.config_version,
        }

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
