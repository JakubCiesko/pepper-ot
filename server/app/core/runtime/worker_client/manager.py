import asyncio
import base64
import contextlib
import logging
from pathlib import Path
import sys
import time
from typing import Any
from uuid import uuid4

import httpx

from app.core.runtime.worker_client.errors import WorkerCircuitOpenError
from app.core.runtime.worker_client.errors import WorkerProtocolError
from app.core.runtime.worker_client.errors import WorkerQueueFullError
from app.core.runtime.worker_client.errors import WorkerStartupTimeoutError
from app.core.runtime.worker_client.errors import WorkerUnavailableError
from app.core.runtime.worker_client.rpc import DetectRPCResponse
from app.core.runtime.worker_client.rpc import WorkerStatusResponse
from app.core.runtime.worker_client.types import RestartReason
from app.core.runtime.worker_client.types import StopReason
from app.core.runtime.worker_client.types import WorkerState
from app.core.runtime.worker_client.types import WorkerStatusSnapshot
from app.schemas.config import AppConfig

logger = logging.getLogger(__name__)


class WorkerManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self._state = WorkerState.STOPPED
        self._process: asyncio.subprocess.Process | None = None
        self._start_lock = asyncio.Lock()
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

    async def start_monitor(self):
        if self._monitor_task and not self._monitor_task.done():
            return
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop_monitor(self):
        if self._monitor_task is None:
            return
        self._monitor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._monitor_task
        self._monitor_task = None

    async def close(self):
        await self.stop(StopReason.SHUTDOWN)
        await self.stop_monitor()
        await self._client.aclose()

    async def update_config(self, config: AppConfig):
        self.config = config

    async def apply_hot_config(self, config: AppConfig, version: int):
        await self.update_config(config)
        self._config_version = version
        if not self.enabled or self._state == WorkerState.STOPPED:
            return
        await self._post_json(
            "/internal/config/hot_update",
            {"config": config.model_dump(mode="json"), "config_version": version},
            timeout=self.internal_timeout,
        )

    async def hard_reload(self, config: AppConfig, version: int):
        await self.update_config(config)
        self._config_version = version
        if not self.enabled:
            await self.stop(StopReason.CONFIG_RELOAD)
            return
        await self.stop(StopReason.CONFIG_RELOAD)

    async def warmup(self):
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
                "deadline_ms": int(self.internal_timeout * 1000),
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
            return out
        finally:
            self._inflight_count = max(0, self._inflight_count - 1)
            if self._state == WorkerState.BUSY:
                self._state = WorkerState.READY

    async def stop(self, reason: StopReason):
        if self._process is None:
            self._state = WorkerState.STOPPED
            await self._cleanup_stream_tasks()
            return
        self._state = WorkerState.STOPPING
        try:
            await self._post_json(
                "/internal/shutdown",
                {"reason": reason.value},
                timeout=self.config.worker.shutdown_grace_seconds,
            )
        except Exception:
            logger.warning("Graceful worker shutdown call failed, forcing termination")
        try:
            await asyncio.wait_for(
                self._process.wait(),
                timeout=self.config.worker.shutdown_grace_seconds,
            )
        except TimeoutError:
            logger.warning("Worker did not exit in grace period; forcing terminate")
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except TimeoutError:
                self._process.kill()
                await self._process.wait()
        await self._cleanup_stream_tasks()
        self._process = None
        self._state = WorkerState.STOPPED
        self._started_at = None

    async def ensure_started(self, reason: RestartReason):
        if not self.enabled:
            raise WorkerUnavailableError("Worker mode is disabled")
        self._check_circuit_breaker()
        if self._process and self._state in {WorkerState.READY, WorkerState.BUSY}:
            return

        if self._state == WorkerState.STARTING:
            if self._startup_waiters >= self.config.worker.max_startup_queue:
                raise WorkerQueueFullError("Worker startup queue is full")
            self._startup_waiters += 1
            try:
                await asyncio.wait_for(
                    self._startup_event.wait(),
                    timeout=self.config.worker.startup_timeout_seconds,
                )
                if self._state not in {WorkerState.READY, WorkerState.BUSY}:
                    raise WorkerStartupTimeoutError(
                        "Worker failed to reach READY state"
                    )
                return
            finally:
                self._startup_waiters = max(0, self._startup_waiters - 1)

        async with self._start_lock:
            if self._process and self._state in {WorkerState.READY, WorkerState.BUSY}:
                return
            await self._start_worker(reason)

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
                logger.warning(f"Failed to get remote worker status: {exc}")
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

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        await self.ensure_started(RestartReason.LAZY_START)
        timeout = timeout or self.internal_timeout
        url = f"{self.base_url}{path}"
        response = await self._client.request(
            method.upper(),
            url,
            json=json,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            self._last_active = time.time()
            return payload
        raise WorkerProtocolError(f"Invalid JSON response from {path}")

    async def _start_worker(self, reason: RestartReason):
        self._state = WorkerState.STARTING
        self._startup_event.clear()
        now = time.time()
        self._restart_timestamps.append(now)
        self._restart_timestamps = [
            ts
            for ts in self._restart_timestamps
            if now - ts <= self.config.worker.restart_window_seconds
        ]
        if len(self._restart_timestamps) > self.config.worker.restart_max_attempts:
            self._breaker_open_until = (
                now + self.config.worker.circuit_breaker_cooldown_seconds
            )
            self._state = WorkerState.FAILED
            raise WorkerCircuitOpenError(
                "Worker restart limit reached; circuit breaker opened"
            )

        logger.info(f"Starting worker process; reason={reason.value}")
        # Run uvicorn from the server root so `app.*` imports resolve.
        cwd = Path(__file__).resolve().parents[4]
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "app.worker.main:app",
            "--host",
            self.config.worker.host,
            "--port",
            str(self.config.worker.port),
            "--log-level",
            "info",
        ]
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self._stdout_task = asyncio.create_task(
            self._forward_stream(self._process.stdout, "stdout")
        )
        self._stderr_task = asyncio.create_task(
            self._forward_stream(self._process.stderr, "stderr")
        )
        self._started_at = time.time()
        self._restart_count += 1

        try:
            await self._wait_until_healthy(self.config.worker.startup_timeout_seconds)
            await self._post_json(
                "/internal/config/reload",
                {
                    "config": self.config.model_dump(mode="json"),
                    "config_version": self._config_version,
                    "deadline_ms": int(
                        self.config.worker.startup_timeout_seconds * 1000
                    ),
                    "request_id": str(uuid4()),
                },
                timeout=self.config.worker.startup_timeout_seconds,
            )
            if self.config.worker.auto_warmup_on_startup:
                await self._post_json(
                    "/internal/warmup",
                    {"config_version": self._config_version},
                    timeout=self.config.worker.startup_timeout_seconds,
                )
            self._state = WorkerState.READY
            self._last_active = time.time()
            self._last_error = None
            self._startup_event.set()
        except Exception as exc:
            self._last_error = str(exc)
            self._state = WorkerState.FAILED
            self._startup_event.set()
            await self.stop(StopReason.FAILURE)
            raise WorkerStartupTimeoutError(str(exc)) from exc

    async def _wait_until_healthy(self, timeout: float):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process and self._process.returncode is not None:
                raise WorkerUnavailableError(
                    f"Worker exited during startup with code {self._process.returncode}"
                )
            try:
                payload = await self._get_json(
                    "/internal/health",
                    timeout=self.config.worker.healthcheck_interval_seconds,
                )
                if payload.get("ok"):
                    return
            except Exception:
                pass
            await asyncio.sleep(self.config.worker.healthcheck_interval_seconds)
        raise WorkerStartupTimeoutError("Worker healthcheck timed out")

    async def _monitor_loop(self):
        try:
            while True:
                await asyncio.sleep(self.config.worker.idle_check_interval_seconds)
                if not self.enabled:
                    continue
                if self._process and self._process.returncode is not None:
                    if self._state not in {WorkerState.STOPPING, WorkerState.STOPPED}:
                        self._state = WorkerState.FAILED
                        self._crash_count += 1
                        self._last_error = (
                            f"Worker crashed with code {self._process.returncode}"
                        )
                        self._process = None
                    continue
                if self._state != WorkerState.READY or self._inflight_count > 0:
                    continue
                idle_for = time.time() - self._last_active
                if idle_for >= self.config.worker.idle_timeout_seconds:
                    self._idle_kill_count += 1
                    logger.info(
                        f"Worker idle timeout reached ({idle_for:.2f}s); stopping worker"
                    )
                    await self.stop(StopReason.IDLE)
        except asyncio.CancelledError:
            return

    def _check_circuit_breaker(self):
        if self._breaker_open_until is None:
            return
        now = time.time()
        if now >= self._breaker_open_until:
            self._breaker_open_until = None
            return
        raise WorkerCircuitOpenError(
            f"Circuit breaker open until {self._breaker_open_until}"
        )

    async def _get_json(self, path: str, timeout: float) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = await self._client.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()

    async def _post_json(
        self, path: str, payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = await self._client.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return data
        raise WorkerProtocolError(f"Invalid worker response for {path}")

    async def _forward_stream(
        self, stream: asyncio.StreamReader | None, name: str
    ) -> None:
        if stream is None:
            return
        pid = self._process.pid if self._process is not None else "?"
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if not text:
                    continue
                prefix = f"[pid={pid} cfg={self._config_version}]"
                if name == "stderr":
                    logger.warning("%s %s", prefix, text)
                else:
                    logger.info("%s %s", prefix, text)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning("Worker %s stream forwarder failed: %s", name, exc)

    async def _cleanup_stream_tasks(self) -> None:
        tasks = [self._stdout_task, self._stderr_task]
        self._stdout_task = None
        self._stderr_task = None
        for task in tasks:
            if task is not None and not task.done():
                task.cancel()
        for task in tasks:
            if task is None:
                continue
            with contextlib.suppress(asyncio.CancelledError):
                await task
