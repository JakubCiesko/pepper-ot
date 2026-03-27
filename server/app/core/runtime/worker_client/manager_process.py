from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
import sys
import time
from uuid import uuid4

from app.core.runtime.worker_client.errors import WorkerCircuitOpenError
from app.core.runtime.worker_client.errors import WorkerStartupTimeoutError
from app.core.runtime.worker_client.errors import WorkerUnavailableError
from app.core.runtime.worker_client.types import RestartReason
from app.core.runtime.worker_client.types import StopReason
from app.core.runtime.worker_client.types import WorkerState

logger = logging.getLogger(__name__)


class WorkerProcessMixin:
    async def _stop_unlocked(self, reason: StopReason):

        if self._process is None or self._state in {
            WorkerState.STOPPED,
            WorkerState.STOPPING,
        }:
            self._state = WorkerState.STOPPED
            await self._cleanup_stream_tasks()
            return
        logger.info("Stopping worker process reason=%s", reason.value)
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

    async def stop(self, reason: StopReason):
        async with self._lifecycle_lock:
            await self._stop_unlocked(reason)

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

        logger.info("Starting worker process reason=%s", reason.value)
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
            logger.info("Worker process is READY pid=%s", self._process.pid)
        except Exception as exc:
            self._last_error = str(exc)
            self._state = WorkerState.FAILED
            self._startup_event.set()
            await self._stop_unlocked(StopReason.FAILURE)
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
