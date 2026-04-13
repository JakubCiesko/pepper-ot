import asyncio
import base64
import logging
import time
from typing import Any
from uuid import uuid4

import httpx

from app.core.runtime.worker_client.errors import WorkerProtocolError
from app.core.runtime.worker_client.manager_monitor import WorkerMonitorMixin
from app.core.runtime.worker_client.manager_process import WorkerProcessMixin
from app.core.runtime.worker_client.manager_rpc import WorkerRPCMixin
from app.core.runtime.worker_client.rpc import DetectRPCResponse
from app.core.runtime.worker_client.rpc import WorkerStatusResponse
from app.core.runtime.worker_client.types import RestartReason
from app.core.runtime.worker_client.types import StopReason
from app.core.runtime.worker_client.types import WorkerState
from app.core.runtime.worker_client.types import WorkerStatusSnapshot
from app.schemas.config import AppConfig

logger = logging.getLogger(__name__)


class WorkerManager(WorkerMonitorMixin, WorkerProcessMixin, WorkerRPCMixin):
    def __init__(self, config: AppConfig):
        self.config = config
        self._state = WorkerState.STOPPED
        self._process: asyncio.subprocess.Process | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._startup_event = asyncio.Event()
        self._startup_waiters = 0
        self._inflight_count = 0
        self._last_active = time.time()
        self._started_at: float | None = None
        self._restart_timestamps: list[float] = []
        self._restart_count = 0
        self._idle_kill_count = 0
        self._crash_count = 0
        self._breaker_open_until: float | None = None
        self._last_error: str | None = None
        self._monitor_task: asyncio.Task | None = None
        self._stdout_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._client = httpx.AsyncClient()
        self._config_version = 0
        logger.info(
            "WorkerManager initialized host=%s port=%s",
            config.worker.host,
            config.worker.port,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.config.worker.enabled)

    @property
    def base_url(self) -> str:
        return f"http://{self.config.worker.host}:{self.config.worker.port}"

    @property
    def internal_timeout(self) -> float:
        return float(self.config.worker.request_timeout_seconds)

    def status_snapshot(self) -> WorkerStatusSnapshot:
        uptime = 0.0
        if self._started_at is not None and self._state != WorkerState.STOPPED:
            uptime = max(0.0, time.time() - self._started_at)
        return WorkerStatusSnapshot(
            state=self._state,
            pid=self._process.pid if self._process else None,
            uptime_seconds=uptime,
            inflight_count=self._inflight_count,
            last_active_ts=self._last_active,
            config_version=self._config_version,
            restart_count=self._restart_count,
            idle_kill_count=self._idle_kill_count,
            crash_count=self._crash_count,
            breaker_open_until=self._breaker_open_until,
            last_error=self._last_error,
            started_at=self._started_at or time.time(),
        )

    async def update_config(self, config: AppConfig):
        self.config = config
        logger.info("WorkerManager config updated")

    async def apply_hot_config(self, config: AppConfig, version: int):
        await self.update_config(config)
        self._config_version = version
        if not self.enabled or self._state == WorkerState.STOPPED:
            logger.info(
                "Skipping worker hot config push enabled=%s state=%s",
                self.enabled,
                self._state,
            )
            return
        logger.info("Pushing hot worker config version=%s", version)
        await self._post_json(
            "/internal/config/hot_update",
            {"config": config.model_dump(mode="json"), "config_version": version},
            timeout=self.internal_timeout,
        )

    async def hard_reload(self, config: AppConfig, version: int):
        async with self._lifecycle_lock:
            await self.update_config(config)
            self._config_version = version
            if not self.enabled:
                await self._stop_unlocked(StopReason.CONFIG_RELOAD)
                return
            logger.info("Applying hard reload marker version=%s", version)
            await self._stop_unlocked(StopReason.CONFIG_RELOAD)

    async def warmup(self):
        logger.info("Worker warmup requested")
        await self.ensure_started(RestartReason.MANUAL_WARMUP)
        await self._post_json(
            "/internal/warmup",
            {"config_version": self._config_version},
            timeout=self.config.worker.startup_timeout_seconds,
        )

    async def detect(self, image_bytes: bytes, robot_metadata) -> dict[str, Any]:
        await self.ensure_started(RestartReason.LAZY_START)
        self._state = WorkerState.BUSY
        self._inflight_count += 1
        started = time.perf_counter()
        try:
            body = {
                "request_id": str(uuid4()),
                "config_version": self._config_version,
                "image_b64": base64.b64encode(image_bytes).decode("utf-8"),
                "robot_metadata": (
                    robot_metadata.model_dump() if robot_metadata else None
                ),
            }
            payload = await self._post_json(
                "/internal/detect", body, timeout=self.internal_timeout
            )
            resp = DetectRPCResponse(**payload)
            if not resp.ok:
                raise WorkerProtocolError(resp.error_message or "worker detect failed")
            self._last_active = time.time()
            out = resp.model_dump(mode="json")
            out["proxy_latency_s"] = time.perf_counter() - started
            logger.info("Worker detect completed latency=%.3fs", out["proxy_latency_s"])
            return out
        finally:
            self._inflight_count = max(0, self._inflight_count - 1)
            if self._state == WorkerState.BUSY:
                self._state = WorkerState.READY

    async def get_worker_status(self) -> WorkerStatusResponse:
        local = self.status_snapshot()
        if self._process and self._state not in {
            WorkerState.STOPPED,
            WorkerState.FAILED,
        }:
            try:
                payload = await self._get_json(
                    "/internal/status",
                    timeout=self.config.worker.healthcheck_interval_seconds,
                )
                remote = WorkerStatusResponse(**payload)
                remote.restart_count = local.restart_count
                remote.idle_kill_count = local.idle_kill_count
                remote.crash_count = local.crash_count
                remote.breaker_open_until = local.breaker_open_until
                remote.last_error = local.last_error
                remote.config_version = local.config_version
                return remote
            except Exception as exc:
                logger.warning("Failed to get remote worker status: %s", exc)
        return WorkerStatusResponse(
            ok=True,
            state=local.state,
            worker_state=local.state,
            pid=local.pid,
            uptime_seconds=local.uptime_seconds,
            inflight_count=local.inflight_count,
            last_active_ts=local.last_active_ts,
            config_version=local.config_version,
            restart_count=local.restart_count,
            idle_kill_count=local.idle_kill_count,
            crash_count=local.crash_count,
            breaker_open_until=local.breaker_open_until,
            last_error=local.last_error,
        )
