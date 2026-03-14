from typing import Any

from pydantic import BaseModel
from pydantic import Field

from app.core.runtime.worker_types import WorkerState
from app.schemas.robot import RobotMetadata


class WorkerRPCRequest(BaseModel):
    request_id: str
    config_version: int
    deadline_ms: int


class WorkerRPCResponse(BaseModel):
    ok: bool = True
    error_code: str | None = None
    error_message: str | None = None
    worker_state: WorkerState = WorkerState.READY
    config_version: int = 0
    elapsed_ms: float = 0.0


class DetectRPCRequest(WorkerRPCRequest):
    image_b64: str
    robot_metadata: RobotMetadata | None = None


class DetectRPCResponse(WorkerRPCResponse):
    image_b64: str | None = None
    objects: list[dict[str, Any]] = Field(default_factory=list)
    scene_graph: list[dict[str, Any]] = Field(default_factory=list)
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
