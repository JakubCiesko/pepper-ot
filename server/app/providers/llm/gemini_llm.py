import logging
from typing import Any

from google.genai import types

from app.providers.common.io import parse_structured_text
from app.providers.common.io import resolve_structured_mode
from app.providers.common.io import schema_to_json_schema
from app.providers.common.utils import normalize_call_kwargs
from app.providers.llm.base import BaseTextProvider
from app.providers.llm.base import LLMResponse
from app.schemas.config import LLMConfig

logger = logging.getLogger(__name__)


class GeminiTextProvider(BaseTextProvider):
    def __init__(self, client):
        self.client = client

    async def generate(
        self,
        *,
        config: LLMConfig,
        messages: list[dict[str, Any]],
        output_schema: Any | None,
        call_overrides: dict[str, Any] | None,
    ) -> LLMResponse:
        kwargs = dict(config.call_kwargs or {})
        if call_overrides:
            kwargs.update(call_overrides)
        kwargs = normalize_call_kwargs(config.provider, kwargs)
        mode = resolve_structured_mode(
            config,
            output_schema=output_schema,
            supports_native_structured=True,
            provider_name="gemini",
        )

        combined_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                combined_parts.extend(
                    [
                        f"{role.upper()}: {item.get('text', '')}"
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ]
                )
            else:
                combined_parts.append(f"{role.upper()}: {content}")
        combined_prompt = "\n\n".join(combined_parts)

        generate_config = kwargs.pop("generate_content_config", {})
        if mode == "provider_native" and output_schema is not None:
            schema_dict = schema_to_json_schema(output_schema)
            if schema_dict is not None:
                generate_config.setdefault("response_mime_type", "application/json")
                generate_config.setdefault("response_json_schema", schema_dict)
            else:
                logger.warning(
                    "Gemini provider_native requested but schema conversion failed; "
                    "continuing with parse_output-compatible response parsing"
                )
        elif mode == "parse_output" and output_schema is not None:
            generate_config.setdefault("response_mime_type", "application/json")

        cfg = types.GenerateContentConfig(**generate_config)
        response = await self.client.aio.models.generate_content(
            model=config.model_id,
            contents=combined_prompt,
            config=cfg,
            **kwargs,
        )

        text = str(getattr(response, "text", "") or "")
        parsed = parse_structured_text(
            text,
            output_schema,
            strict=config.structured_output.strict,
        )
        return LLMResponse(text=text, parsed=parsed, raw=response)
