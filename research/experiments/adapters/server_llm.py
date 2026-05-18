from typing import Any

from .bootstrap import ensure_server_app_importable


class ServerLLMAdapter:
    """In-process adapter around the server LLM client.

    The research workflows use this adapter to reuse the same provider,
    structured-output, and model configuration code as the server without
    starting an HTTP service.
    """

    def __init__(
        self,
        provider: str,
        model_id: str,
        structured_mode: str = "provider_native",
        base_url: str | None = None,
    ):
        """Create an LLM adapter from research model config fields.

        Args:
            provider: Server LLM provider name.
            model_id: Provider-specific model identifier.
            structured_mode: Structured output strategy used by the server
                LLMClient.
            base_url: Optional OpenAI-compatible endpoint base URL.
        """
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
        """Generate unstructured text from system and user prompts."""
        return await self._client.generate_text(system_prompt, user_prompt)

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: Any,
    ) -> Any:
        """Generate a structured response validated against an output schema.

        Args:
            system_prompt: System prompt sent to the LLM provider.
            user_prompt: User prompt sent to the LLM provider.
            output_schema: Pydantic model or schema object expected by the
                server LLM client.

        Returns:
            Provider response object from LLMClient.generate, including parsed
            structured output when available.
        """
        resp = await self._client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=output_schema,
        )
        return resp
