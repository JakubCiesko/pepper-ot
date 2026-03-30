import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from app.providers.common.io import parse_structured_text
from app.providers.common.utils import normalize_call_kwargs
from app.providers.llm.base import BaseTextProvider
from app.providers.llm.base import LLMResponse
from app.schemas.config import LLMConfig

logger = logging.getLogger(__name__)


class LocalHFTextProvider(BaseTextProvider):
    def __init__(self, config: LLMConfig):
        self.config = config
        requested_device = config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        dtype = torch.bfloat16 if requested_device.startswith("cuda") else torch.float32

        model_kwargs = dict(config.client_init_kwargs or {})
        model_kwargs.setdefault("trust_remote_code", True)
        model_kwargs.setdefault("dtype", dtype)
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
        kwargs = normalize_call_kwargs(config.provider, kwargs)

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
