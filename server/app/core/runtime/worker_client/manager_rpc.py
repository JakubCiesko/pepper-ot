from __future__ import annotations

import time
from typing import Any

from app.core.runtime.worker_client.errors import WorkerProtocolError
from app.core.runtime.worker_client.types import RestartReason


class WorkerRPCMixin:
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
