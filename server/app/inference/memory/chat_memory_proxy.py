import base64
import time

from app.core.runtime.worker_client.manager import WorkerManager
from app.schemas.scene import SceneState


class EmptyChatMemory:
    """Minimal memory adapter for chat when visual memory lives in worker process."""

    def scene_state(self) -> SceneState:
        return SceneState(objects=[], relationships=[], timestamp=time.time())

    async def get_track_crop(self, object_id: int) -> bytes | None:
        return None


class WorkerChatMemoryProxy:
    """Memory adapter for chat that reads authoritative state from worker process."""

    def __init__(self, worker_manager: WorkerManager):
        self.worker_manager = worker_manager

    async def scene_state(self) -> SceneState:
        try:
            payload = await self.worker_manager.request("GET", "/internal/memory")
            return SceneState(**payload)
        except Exception:
            return SceneState(objects=[], relationships=[], timestamp=time.time())

    async def get_track_crop(self, object_id: int) -> bytes | None:
        try:
            payload = await self.worker_manager.request(
                "GET", f"/internal/memory/object/{object_id}/crop"
            )
            image_b64 = payload.get("image_b64")
            if not isinstance(image_b64, str) or not image_b64:
                return None
            return base64.b64decode(image_b64)
        except Exception:
            return None
