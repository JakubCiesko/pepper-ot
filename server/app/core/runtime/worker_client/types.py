from enum import StrEnum
import time

from pydantic import BaseModel
from pydantic import Field


class WorkerState(StrEnum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    READY = "READY"
    BUSY = "BUSY"
    # DRAINING = "DRAINING"
    STOPPING = "STOPPING"
    FAILED = "FAILED"


class RestartReason(StrEnum):
    LAZY_START = "lazy_start"
    CONFIG_RELOAD = "config_reload"
    # CRASH_RECOVERY = "crash_recovery"
    MANUAL_WARMUP = "manual_warmup"


class StopReason(StrEnum):
    IDLE = "idle"
    MANUAL = "manual"
    SHUTDOWN = "shutdown"
    CONFIG_RELOAD = "config_reload"
    FAILURE = "failure"


class WorkerStatusSnapshot(BaseModel):
    state: WorkerState = WorkerState.STOPPED
    pid: int | None = None
    uptime_seconds: float = 0.0
    inflight_count: int = 0
    last_active_ts: float | None = None
    config_version: int = 0
    restart_count: int = 0
    idle_kill_count: int = 0
    crash_count: int = 0
    breaker_open_until: float | None = None
    last_error: str | None = None
    started_at: float = Field(default_factory=time.time)
