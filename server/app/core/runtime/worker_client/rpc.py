from typing import Any

from pydantic import BaseModel
from pydantic import Field

from app.core.runtime.worker_client.types import WorkerState
from app.schemas.robot import RobotMetadata


class WorkerRPCRequest(BaseModel):
    request_id: str
    config_version: int


class WorkerRPCResponse(BaseModel):
    ok: bool = True
    error_message: str | None = None
    worker_state: WorkerState = WorkerState.READY
    config_version: int = 0


class DetectRPCRequest(WorkerRPCRequest):
    image_b64: str
    robot_metadata: RobotMetadata | None = None


class DetectRPCResponse(WorkerRPCResponse):
    image_b64: str | None = None
    objects: list[dict[str, Any]] = Field(default_factory=list)
    scene_graph: list[dict[str, Any]] = Field(default_factory=list)
    caption: str | None = None
    caption_provider: str | None = None
    caption_model_id: str | None = None
    memory: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    executed_stages: list[str] = Field(default_factory=list)
    image_width: int = 0
    image_height: int = 0


class WorkerConfigRPCRequest(WorkerRPCRequest):
    config: dict[str, Any]


class WorkerStatusResponse(WorkerRPCResponse):
    state: WorkerState
    pid: int | None = None
    uptime_seconds: float = 0.0
    inflight_count: int = 0
    last_active_ts: float | None = None
    restart_count: int = 0
    idle_kill_count: int = 0
    crash_count: int = 0
    breaker_open_until: float | None = None
    last_error: str | None = None
