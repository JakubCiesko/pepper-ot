import logging
from typing import Any

import instructor
from openai import AsyncOpenAI

from app.providers.common.io import extract_text_content
from app.providers.common.io import extract_text_from_openai_response
from app.providers.common.io import parse_structured_text
from app.providers.common.io import resolve_structured_mode
from app.providers.common.io import validate_parsed_output
from app.providers.common.utils import normalize_call_kwargs
from app.providers.common.utils import normalize_openai_parse_kwargs
from app.providers.llm.base import BaseTextProvider
from app.providers.llm.base import LLMResponse
from app.schemas.config import LLMConfig

logger = logging.getLogger(__name__)


class OpenAITextProvider(BaseTextProvider):
    def __init__(self, client: AsyncOpenAI, supports_native_structured: bool):
        self.client = client
        self.supports_native_structured = supports_native_structured
        # TODO: Test whether JSON works every time. Tools does not work for small models it seems (without tool calls)
        self.instructor_client = instructor.from_openai(
            self.client,
            mode=instructor.Mode.JSON,  # instructor.Mode.TOOLS #TODO: ERROR:  WARNING: instructor.Mode.JSON for some models without tool calling!
        )

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
            supports_native_structured=self.supports_native_structured,
            provider_name="openai",
        )

        if mode == "provider_native" and output_schema is not None:
            parse_kwargs = normalize_openai_parse_kwargs(kwargs)
            try:
                response = await self.client.responses.parse(
                    model=config.model_id,
                    input=messages,  # type: ignore TODO: look at this type
                    text_format=output_schema,
                    **parse_kwargs,
                )
                parsed_native = getattr(response, "output_parsed", None)
                text = extract_text_from_openai_response(response)
                parsed = validate_parsed_output(
                    parsed_native,
                    output_schema,
                    strict=config.structured_output.strict,
                )
                return LLMResponse(text=text, parsed=parsed, raw=response)
            except Exception as exc:
                logger.warning(
                    "OpenAI provider_native structured call failed, deterministically "
                    "falling back to parse_output provider=%s model=%s error=%s",
                    config.provider,
                    config.model_id,
                    exc,
                )
        if mode == "instructor" and output_schema is not None:
            try:
                parsed = await self.instructor_client.chat.completions.create(
                    model=config.model_id,
                    messages=messages,
                    response_model=output_schema,
                    **kwargs,
                )
                text = (
                    parsed.model_dump_json()
                    if hasattr(parsed, "model_dump_json")
                    else str(parsed)
                )
                return LLMResponse(text=text, parsed=parsed, raw=text)

            except Exception as exc:
                logger.warning("Instructor failed → fallback: %s", exc)
                mode = "parse_output"

        if mode == "parse_output" and output_schema is not None:
            kwargs.setdefault("response_format", {"type": "json_object"})

        response = await self.client.chat.completions.create(
            model=config.model_id,
            messages=messages,
            **kwargs,
        )
        content = response.choices[0].message.content if response.choices else ""
        text = extract_text_content(content)
        parsed = parse_structured_text(
            text,
            output_schema,
            strict=config.structured_output.strict,
        )
        return LLMResponse(text=text, parsed=parsed, raw=response)
