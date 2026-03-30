from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any

from app.providers.common.utils import normalize_call_kwargs
from app.schemas.config import LLMConfig


class BaseVLMClient(ABC):
    def update_runtime(self, config: LLMConfig):
        self.config = config

    def prepare_kwargs(self, call_overrides: dict[str, Any]) -> dict[str, Any]:
        kwargs = dict(self.config.call_kwargs or {})
        if call_overrides:
            kwargs.update(call_overrides)
        return normalize_call_kwargs(self.config.provider, kwargs)

    @abstractmethod
    def prepare_input(
        self,
        system_prompt: str,
        user_prompt: str,
        image: bytes | None,
        output_schema: Any | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def infer(
        self,
        system_prompt: str,
        user_prompt: str,
        image: bytes | None,
        *,
        output_schema: Any | None = None,
        call_overrides: dict[str, Any] | None = None,
    ) -> tuple[str, Any | None]:
        raise NotImplementedError
