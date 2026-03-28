import io
import logging
from typing import Any

from PIL import Image
import torch
from transformers import BlipForConditionalGeneration
from transformers import BlipProcessor

from app.core.config.llm_contracts import normalize_call_kwargs
from app.providers.vlm_client import BaseVLMClient
from app.providers.vlm_client import build_vlm_client
from app.schemas.config import CaptionConfig

logger = logging.getLogger(__name__)


class LocalBLIPCaptionClient:
    def __init__(self, config: CaptionConfig):
        self.config = config
        requested_device = config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device = torch.device(requested_device)

        model_kwargs = dict(config.client_init_kwargs or {})
        processor_kwargs = dict(model_kwargs.pop("processor_kwargs", {}))
        if requested_device.startswith("cuda"):
            model_kwargs.setdefault("dtype", torch.float16)

        self.processor = BlipProcessor.from_pretrained(
            config.model_id, **processor_kwargs, backend="torchvision"  # use_fast=True
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            config.model_id, **model_kwargs
        ).to(self.device)
        self.model.eval()

    # TODO: WORK ON THIS MAINLY INFER, STUPID
    def update_runtime(self, config: CaptionConfig):
        self.config = config

    async def infer(
        self,
        system_prompt: str,
        user_prompt: str,
        image: bytes,
        *,
        call_overrides: dict[str, Any] | None = None,
    ) -> str:
        kwargs = dict(self.config.call_kwargs or {})
        if call_overrides:
            kwargs.update(call_overrides)
        kwargs = normalize_call_kwargs(self.config.provider, kwargs)

        max_new_tokens = int(kwargs.pop("max_new_tokens", kwargs.pop("max_tokens", 64)))
        do_sample = bool(kwargs.pop("do_sample", False))
        temperature = kwargs.pop("temperature", None)
        top_p = kwargs.pop("top_p", None)
        top_k = kwargs.pop("top_k", None)

        # BLIP captioning is prefix-conditioned, not instruction-chat driven.
        # Avoid passing long system instructions that cause prompt echo.
        prompt = " ".join((user_prompt or "").split()).strip()
        if self.config.mode == "unconditional":
            prompt = ""
        elif not prompt or "?" in prompt or len(prompt.split()) > 12:
            prompt = "a photo of"

        img = Image.open(io.BytesIO(image)).convert("RGB")
        if prompt:
            inputs = self.processor(img, prompt, return_tensors="pt")
        else:
            inputs = self.processor(img, return_tensors="pt")
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample and temperature is not None:
            generate_kwargs["temperature"] = float(temperature)
        if do_sample and top_p is not None:
            generate_kwargs["top_p"] = float(top_p)
        if do_sample and top_k is not None:
            generate_kwargs["top_k"] = int(top_k)

        with torch.no_grad():
            output = self.model.generate(**inputs, **generate_kwargs)
        text = self.processor.decode(output[0], skip_special_tokens=True)
        return text.strip()


class CaptionClient:
    def __init__(self, config: CaptionConfig):
        self.config = config
        self._client = self._build_client(config)

    def _build_client(
        self, config: CaptionConfig
    ) -> BaseVLMClient | LocalBLIPCaptionClient:
        if (
            config.provider == "local_hf"
            and "blip-image-captioning" in config.model_id.lower()
        ):
            logger.info("Using BLIP caption client model=%s", config.model_id)
            return LocalBLIPCaptionClient(config)
        return build_vlm_client(config)

    def update_runtime(self, config: CaptionConfig, rebuild_client: bool = False):
        self.config = config
        if rebuild_client:
            self._client = self._build_client(config)
        else:
            self._client.update_runtime(config)

    async def infer(
        self,
        system_prompt: str,
        user_prompt: str,
        image: bytes,
        *,
        call_overrides: dict[str, Any] | None = None,
    ) -> str:
        if isinstance(self._client, LocalBLIPCaptionClient):
            return await self._client.infer(
                system_prompt,
                user_prompt,
                image,
                call_overrides=call_overrides,
            )

        raw, _ = await self._client.infer(
            system_prompt,
            user_prompt,
            image,
            output_schema=None,
            call_overrides=call_overrides,
        )
        return raw.strip()
