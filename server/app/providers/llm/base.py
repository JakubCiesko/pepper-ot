from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from app.schemas.config import LLMConfig


@dataclass
class LLMResponse:
    text: str
    parsed: Any | None = None
    raw: Any | None = None


class BaseTextProvider:
    @abstractmethod
    async def generate(
        self,
        *,
        config: LLMConfig,
        messages: list[dict[str, Any]],
        output_schema: Any | None,
        call_overrides: dict[str, Any] | None,
    ) -> LLMResponse:
        raise NotImplementedError
