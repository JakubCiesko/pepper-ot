from __future__ import annotations

from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any

from app.orchestration.services.memory import DomainNotFoundError
from app.orchestration.services.memory import DomainValidationError
from fastapi import HTTPException


async def run_memory_action(
    action: Callable[[], Awaitable[Any]],
    *,
    on_success: Callable[[], Awaitable[None]] | None = None,
    include_internal_errors: bool = True,
) -> Any:
    try:
        result = await action()
        if on_success is not None:
            await on_success()
        return result
    except DomainValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DomainNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        if include_internal_errors:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        raise
