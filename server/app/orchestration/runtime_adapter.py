import base64
import io
import time
from typing import Any

import numpy as np
from PIL import Image

from app.core.runtime.state import AppState
from app.core.runtime.worker_client.manager import WorkerManager
from app.schemas.robot import RobotMetadata
from app.schemas.scene import Relationship
from app.schemas.scene import SceneState
from app.schemas.scene import TrackedObjectState


class LocalRuntimeAdapter:
    def __init__(self, state: AppState):
        self.state = state

    async def detect(
        self, image_bytes: bytes, metadata: RobotMetadata
    ) -> dict[str, Any]:
        if self.state.pipeline is None:
            raise RuntimeError("Pipeline not initialized")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = await self.state.pipeline.process(image, metadata)

        som = result.som_image if result.som_image is not None else np.array(image)
        som_pil = Image.fromarray(som.astype("uint8"))
        buf = io.BytesIO()
        som_pil.save(buf, format="JPEG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        memory = self.state.pipeline.memory.scene_state().model_dump(mode="json")
        objects = [
            {
                "label": det.label,
                "confidence": det.confidence,
                "bbox": det.bbox,
                "object_id": det.object_id,
            }
            for det in result.detections
        ]

        return {
            "ok": True,
            "image_b64": image_b64,
            "objects": objects,
            "scene_graph": result.scene_graph.as_dict() if result.scene_graph else [],
            "caption": result.caption,
            "caption_provider": result.caption_provider,
            "caption_model_id": result.caption_model_id,
            "memory": memory,
            "metrics": result.metrics,
            "executed_stages": result.executed_stages,
            "image_width": image.width,
            "image_height": image.height,
            "worker_state": "READY",
            "config_version": self.state.config_version,
        }

    async def vision_chat(
        self,
        image_bytes: bytes,
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        if self.state.pipeline is None:
            raise RuntimeError("Pipeline not initialized")
        backend = self.state.pipeline.scene_graph_service.vlm_backend
        system = system_prompt if system_prompt is not None else backend.system_prompt
        text, _parsed = await backend.client.infer(
            system,
            user_prompt,
            image_bytes,
            output_schema=None,
        )
        return {
            "ok": True,
            "answer": text,
            "provider": backend.config.provider,
            "model_id": backend.config.model_id,
        }

    def _memory(self):
        if self.state.pipeline is None or self.state.pipeline.memory is None:
            raise RuntimeError("Memory not initialized")
        return self.state.pipeline.memory

    async def scene_state(self) -> SceneState:
        return self._memory().scene_state()

    async def upsert_scene_state(self, state: SceneState):
        self._memory().upsert_scene_state(state)

    async def reset_memory(self):
        self._memory().reset()

    async def create_object(self, obj: TrackedObjectState):
        self._memory().create_object(obj)

    async def patch_object(self, object_id: int, updates: dict):
        return self._memory().patch_object(object_id, updates)

    async def delete_object(self, object_id: int, cascade_relations: bool):
        return self._memory().delete_object(
            object_id, cascade_relations=cascade_relations
        )

    async def create_relation(self, rel: Relationship):
        self._memory().create_relation(rel)

    async def patch_relation(
        self, subject_id: int, predicate: str, object_id: int, updates: dict
    ):
        return self._memory().patch_relation(subject_id, predicate, object_id, updates)

    async def delete_relation(self, subject_id: int, predicate: str, object_id: int):
        return self._memory().delete_relation(subject_id, predicate, object_id)

    async def next_object_id(self) -> int:
        return self._memory().next_id


class WorkerRuntimeAdapter:
    def __init__(self, worker_manager: WorkerManager):
        self.worker_manager = worker_manager

    async def detect(
        self, image_bytes: bytes, metadata: RobotMetadata
    ) -> dict[str, Any]:
        return await self.worker_manager.detect(image_bytes, metadata)

    async def vision_chat(
        self,
        image_bytes: bytes,
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        payload = await self.worker_manager.request(
            "POST",
            "/internal/vision_chat",
            json={
                "image_b64": base64.b64encode(image_bytes).decode("utf-8"),
                "user_prompt": user_prompt,
                "system_prompt": system_prompt,
            },
        )
        return payload

    async def scene_state(self) -> SceneState:
        payload = await self.worker_manager.request("GET", "/internal/memory")
        return SceneState(**payload)

    async def upsert_scene_state(self, state: SceneState):
        await self.worker_manager.request(
            "POST",
            "/internal/memory/upsert",
            json=state.model_dump(mode="json"),
        )

    async def reset_memory(self):
        await self.worker_manager.request(
            "POST",
            "/internal/memory/reset",
            params={"confirm": True},
        )

    async def create_object(self, obj: TrackedObjectState):
        await self.worker_manager.request(
            "POST",
            "/internal/memory/object",
            json=obj.model_dump(mode="json"),
        )

    async def patch_object(self, object_id: int, updates: dict):
        payload = await self.worker_manager.request(
            "PATCH",
            f"/internal/memory/object/{object_id}",
            json=updates,
        )
        return TrackedObjectState(**payload["object"])

    async def delete_object(self, object_id: int, cascade_relations: bool):
        payload = await self.worker_manager.request(
            "DELETE",
            f"/internal/memory/object/{object_id}",
            params={"cascade_relations": cascade_relations},
        )
        return bool(payload.get("ok", False))

    async def create_relation(self, rel: Relationship):
        await self.worker_manager.request(
            "POST",
            "/internal/memory/relation",
            json=rel.model_dump(mode="json"),
        )

    async def patch_relation(
        self, subject_id: int, predicate: str, object_id: int, updates: dict
    ):
        payload = await self.worker_manager.request(
            "PATCH",
            "/internal/memory/relation",
            json={
                "subject_id": subject_id,
                "predicate": predicate,
                "object_id": object_id,
                **updates,
            },
        )
        return Relationship(**payload["relationship"])

    async def delete_relation(self, subject_id: int, predicate: str, object_id: int):
        payload = await self.worker_manager.request(
            "DELETE",
            "/internal/memory/relation",
            params={
                "subject_id": subject_id,
                "predicate": predicate,
                "object_id": object_id,
            },
        )
        return bool(payload.get("ok", False))

    async def next_object_id(self) -> int:
        state = await self.scene_state()
        if not state.objects:
            return 1
        return max(o.id for o in state.objects) + 1


class WorkerProcessRuntimeAdapter:
    """Adapter used inside the worker process for internal routes."""

    def __init__(self, runtime):
        self.runtime = runtime

    async def scene_state(self) -> SceneState:
        return await self.runtime.scene_state()

    async def upsert_scene_state(self, state: SceneState):
        await self.runtime.upsert_scene_state(state)

    async def reset_memory(self):
        await self.runtime.reset_memory()

    async def create_object(self, obj: TrackedObjectState):
        await self.runtime.create_object(obj)

    async def patch_object(self, object_id: int, updates: dict):
        return await self.runtime.patch_object(object_id, updates)

    async def delete_object(self, object_id: int, cascade_relations: bool):
        return await self.runtime.delete_object(object_id, cascade_relations)

    async def create_relation(self, rel: Relationship):
        await self.runtime.create_relation(rel)

    async def patch_relation(
        self, subject_id: int, predicate: str, object_id: int, updates: dict
    ):
        return await self.runtime.patch_relation(
            subject_id, predicate, object_id, updates
        )

    async def delete_relation(self, subject_id: int, predicate: str, object_id: int):
        return await self.runtime.delete_relation(subject_id, predicate, object_id)

    async def next_object_id(self) -> int:
        await self.runtime.ensure_pipeline()
        return self.runtime.pipeline.memory.next_id


def resolve_runtime_adapter(state: AppState):
    use_worker = bool(
        state.config
        and state.config.worker.enabled
        and state.worker_manager is not None
    )
    if use_worker:
        return WorkerRuntimeAdapter(state.worker_manager)
    return LocalRuntimeAdapter(state)


def memory_payload(state: SceneState | None = None) -> dict[str, Any]:
    return (
        state.model_dump(mode="json")
        if state is not None
        else {"objects": [], "relationships": [], "timestamp": time.time()}
    )
