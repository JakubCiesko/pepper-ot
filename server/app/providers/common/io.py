import json
import logging
from typing import Any
from typing import Literal

from pydantic import TypeAdapter

from app.providers.common.utils import provider_capability_matrix
from app.schemas.config import LLMConfig

logger = logging.getLogger(__name__)

StructuredMode = Literal["provider_native", "parse_output", "instructor"]


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(filter(None, parts))
    return str(content or "")


def extract_json_block(raw: str) -> str | None:
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return None


def parse_structured_text(raw: str, schema: Any | None, strict: bool) -> Any | None:
    if schema is None:
        return None
    block = extract_json_block(raw)
    if block is None:
        if strict:
            raise ValueError("Structured output requested but no JSON found")
        return None
    parsed = json.loads(block)
    adapter = TypeAdapter(schema)
    return adapter.validate_python(parsed)


def validate_parsed_output(parsed: Any, schema: Any | None, strict: bool) -> Any | None:
    if schema is None:
        return None
    if parsed is None:
        if strict:
            raise ValueError("Structured output is empty")
        return None
    adapter = TypeAdapter(schema)
    return adapter.validate_python(parsed)


def schema_to_json_schema(schema: Any | None) -> dict[str, Any] | None:
    if schema is None:
        return None
    try:
        return TypeAdapter(schema).json_schema()
    except Exception:
        return None


def resolve_structured_mode(
    config: LLMConfig,
    *,
    output_schema: Any | None,
    supports_native_structured: bool,
    provider_name: str,
) -> StructuredMode:
    if output_schema is None:
        return "parse_output"

    capability_matrix = provider_capability_matrix().get(
        "structured_output_support", {}
    )
    provider_key = str(config.provider)
    provider_caps = capability_matrix.get(provider_key, {})

    supports_native_from_matrix = bool(provider_caps.get("provider_native", False))
    supports_instructor_from_matrix = bool(provider_caps.get("instructor", False))

    # Keep this parameter for compatibility at call sites; matrix remains source of truth.
    if (
        supports_native_structured != supports_native_from_matrix
        and provider_key in capability_matrix
    ):
        logger.warning(
            "Structured native support mismatch provider=%s provider_name=%s callsite=%s matrix=%s; using matrix",
            provider_key,
            provider_name,
            supports_native_structured,
            supports_native_from_matrix,
        )

    mode = config.structured_output.mode
    if mode == "provider_native" and not supports_native_from_matrix:
        logger.warning(
            "Structured mode provider_native requested for provider=%s provider_name=%s but not supported; using parse_output",
            provider_key,
            provider_name,
        )
        return "parse_output"
    if mode == "instructor" and not supports_instructor_from_matrix:
        logger.warning(
            "Structured mode instructor requested for provider=%s provider_name=%s but not supported; using parse_output",
            provider_key,
            provider_name,
        )
        return "parse_output"
    return mode


def extract_text_from_openai_response(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text:
        return text
    output = getattr(response, "output", None)
    if not output:
        return ""
    parts: list[str] = []
    for item in output:
        content = getattr(item, "content", None) or []
        for chunk in content:
            chunk_text = getattr(chunk, "text", None)
            if chunk_text:
                parts.append(str(chunk_text))
    return "\n".join(parts)


# todo: prompt tools one source of truth, for both system, user, {context}, {caption}, {last:captions}
