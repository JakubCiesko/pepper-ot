from __future__ import annotations

import asyncio
import contextlib
import logging
import time

from app.core.runtime.worker_client.errors import WorkerCircuitOpenError
from app.core.runtime.worker_client.errors import WorkerQueueFullError
from app.core.runtime.worker_client.errors import WorkerStartupTimeoutError
from app.core.runtime.worker_client.errors import WorkerUnavailableError
from app.core.runtime.worker_client.types import RestartReason
from app.core.runtime.worker_client.types import StopReason
from app.core.runtime.worker_client.types import WorkerState

logger = logging.getLogger(__name__)


class WorkerMonitorMixin:
    async def start_monitor(self):
        if self._monitor_task and not self._monitor_task.done():
            return
        logger.info("Starting worker monitor loop")
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop_monitor(self):
        if self._monitor_task is None:
            return
        logger.info("Stopping worker monitor loop")
        self._monitor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._monitor_task
        self._monitor_task = None

    async def close(self):
        logger.info("Closing WorkerManager resources")
        await self.stop(StopReason.SHUTDOWN)
        await self.stop_monitor()
        await self._client.aclose()

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
                        logger.warning(self._last_error)
                        self._process = None
                    continue
                if self._state != WorkerState.READY or self._inflight_count > 0:
                    continue
                idle_for = time.time() - self._last_active
                if idle_for >= self.config.worker.idle_timeout_seconds:
                    self._idle_kill_count += 1
                    logger.info(
                        "Worker idle timeout reached (%.2fs); stopping worker", idle_for
                    )
                    await self.stop(StopReason.IDLE)
        except asyncio.CancelledError:
            return

    def _check_circuit_breaker(self):
        if self._breaker_open_until is None:
            return
        now = time.time()
        if now >= self._breaker_open_until:
            logger.info("Worker circuit breaker cooldown elapsed")
            self._breaker_open_until = None
            return
        raise WorkerCircuitOpenError(
            f"Circuit breaker open until {self._breaker_open_until}"
        )
