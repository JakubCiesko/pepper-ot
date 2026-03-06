from dataclasses import dataclass
import logging
from typing import Any

from google import genai
from google.genai import types  # type: ignore
from openai import AsyncOpenAI
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from app.schemas.config import LLMConfig
from app.services.model_io_common import extract_text_content
from app.services.model_io_common import extract_text_from_openai_response
from app.services.model_io_common import parse_structured_text
from app.services.model_io_common import resolve_structured_mode
from app.services.model_io_common import schema_to_json_schema
from app.services.model_io_common import validate_parsed_output
from app.services.provider_runtime import build_gemini_client_kwargs
from app.services.provider_runtime import build_openai_async_client_kwargs

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    parsed: Any | None = None
    raw: Any | None = None


class BaseTextProvider:
    async def generate(
        self,
        *,
        config: LLMConfig,
        messages: list[dict[str, Any]],
        output_schema: Any | None,
        call_overrides: dict[str, Any] | None,
    ) -> LLMResponse:
        raise NotImplementedError


class OpenAITextProvider(BaseTextProvider):
    def __init__(self, client: AsyncOpenAI, supports_native_structured: bool):
        self.client = client
        self.supports_native_structured = supports_native_structured

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

        mode = resolve_structured_mode(
            config,
            output_schema=output_schema,
            supports_native_structured=self.supports_native_structured,
            provider_name="openai",
        )

        if mode == "provider_native" and output_schema is not None:
            parse_kwargs = dict(kwargs)
            if "max_tokens" in parse_kwargs and "max_output_tokens" not in parse_kwargs:
                parse_kwargs["max_output_tokens"] = parse_kwargs.pop("max_tokens")
            try:
                response = await self.client.responses.parse(
                    model=config.model_id,
                    input=messages,
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
                    "OpenAI provider_native structured call failed, falling back to parse_output: %s",
                    exc,
                )

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


class LocalHFTextProvider(BaseTextProvider):
    def __init__(self, config: LLMConfig):
        self.config = config
        requested_device = config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        dtype = torch.bfloat16 if requested_device.startswith("cuda") else torch.float32

        model_kwargs = dict(config.client_init_kwargs or {})
        model_kwargs.setdefault("trust_remote_code", True)
        model_kwargs.setdefault("torch_dtype", dtype)
        if "device_map" not in model_kwargs:
            model_kwargs["device_map"] = (
                "auto" if requested_device.startswith("cuda") else {"": "cpu"}
            )

        tokenizer_kwargs = dict(model_kwargs.pop("tokenizer_kwargs", {}))
        tokenizer_kwargs.setdefault("trust_remote_code", True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, **tokenizer_kwargs
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id, **model_kwargs
        )
        self.device = (
            self.model.device
            if hasattr(self.model, "device")
            else torch.device(requested_device)
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

        max_new_tokens = int(
            kwargs.pop("max_new_tokens", kwargs.pop("max_tokens", 256))
        )
        do_sample = bool(kwargs.pop("do_sample", True))
        temperature = float(kwargs.pop("temperature", 0.7 if do_sample else 1.0))
        top_p = float(kwargs.pop("top_p", 1.0))

        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            prompt_parts.append(f"{role.upper()}: {msg.get('content', '')}")
        prompt = "\\n\\n".join(prompt_parts) + "\\n\\nASSISTANT:"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )

        trimmed = out[:, inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0]
        parsed = parse_structured_text(
            text,
            output_schema,
            strict=config.structured_output.strict,
        )
        return LLMResponse(text=text, parsed=parsed, raw=None)


class LLMClient:
    """
    Provider-agnostic text generation client.

    Supports OpenAI, Gemini, and OpenAI-compatible endpoints (vLLM/Ollama-like APIs).
    Runtime config updates are supported via update_runtime().
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider: BaseTextProvider | None = None
        self.update_runtime(config, rebuild_client=True)

    def update_runtime(self, config: LLMConfig, rebuild_client: bool = True):
        self.config = config
        if rebuild_client or self.provider is None:
            self.provider = self._build_provider(config)

    def _build_provider(self, config: LLMConfig) -> BaseTextProvider:
        provider = config.provider

        if provider in {"openai", "openai_compatible"}:
            client_kwargs = build_openai_async_client_kwargs(
                config,
                default_api_env="OPENAI_API_KEY",
                default_api_value="EMPTY",
            )
            return OpenAITextProvider(
                AsyncOpenAI(**client_kwargs),
                supports_native_structured=(provider == "openai"),
            )

        if provider in {"local_hf", "local_4bit"}:
            return LocalHFTextProvider(config)

        if provider == "gemini":
            client_kwargs = build_gemini_client_kwargs(
                config,
                default_api_env="GEMINI_API_KEY",
                default_api_value="",
            )
            return GeminiTextProvider(genai.Client(**client_kwargs))

        raise ValueError(f"Unsupported LLM provider: {provider}")

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        output_schema: Any | None = None,
        call_overrides: dict[str, Any] | None = None,
    ) -> LLMResponse:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self.provider is None:
            self.update_runtime(self.config, rebuild_client=True)
        return await self.provider.generate(
            config=self.config,
            messages=messages,
            output_schema=output_schema,
            call_overrides=call_overrides,
        )

    async def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = await self.generate(system_prompt, user_prompt)
            return response.text or "I'm not sure what to say."
        except Exception as exc:
            logger.error(f"LLM generation error: {exc}")
            return "I am having trouble connecting to my language center right now."
