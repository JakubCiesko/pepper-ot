import time

from app.core.runtime.worker_client.manager import WorkerManager
from app.schemas.scene import SceneState


class EmptyChatMemory:
    """Minimal memory adapter for chat when visual memory lives in worker process."""

    def scene_state(self) -> SceneState:
        return SceneState(objects=[], relationships=[], timestamp=time.time())


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
