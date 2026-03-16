from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any

from app.schemas.config import LLMConfig


class BaseVLMClient(ABC):
    def update_runtime(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    async def infer(
        self,
        system_prompt: str,
        user_prompt: str,
        image: bytes,
        *,
        output_schema: Any | None = None,
        call_overrides: dict[str, Any] | None = None,
    ) -> tuple[str, Any | None]:
        raise NotImplementedError
