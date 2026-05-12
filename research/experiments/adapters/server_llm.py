from typing import Any

from .bootstrap import ensure_server_app_importable


class ServerLLMAdapter:
    def __init__(
        self,
        provider: str,
        model_id: str,
        structured_mode: str = "provider_native",
        base_url: str | None = None,
    ):
        ensure_server_app_importable()
        from app.providers.llm.client import LLMClient
        from app.schemas.config import LLMConfig
        from app.schemas.config import StructuredOutputConfig

        cfg = LLMConfig(
            provider=provider,
            model_id=model_id,
            structured_output=StructuredOutputConfig(mode=structured_mode),
            base_url=base_url,
        )
        self._client = LLMClient(cfg)

    async def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        return await self._client.generate_text(system_prompt, user_prompt)

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: Any,
    ) -> Any:
        resp = await self._client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=output_schema,
        )
        return resp
