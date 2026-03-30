from dataclasses import dataclass
import io
import logging
from typing import Any

from PIL import Image

from app.providers.caption.client import CaptionClient
from app.providers.common.utils import normalize_call_kwargs
from app.schemas.config import CaptionConfig

logger = logging.getLogger(__name__)


@dataclass
class CaptionInferenceResult:
    text: str
    provider: str
    model_id: str


class CaptionInferenceService:
    """Pure inference caption service (no websocket/API side effects)."""

    def __init__(
        self,
        config: CaptionConfig,
        *,
        system_prompt: str,
        user_prompt: str | None,
    ):
        self.config = config
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.client = CaptionClient(config)

    def update_runtime(
        self,
        config: CaptionConfig,
        *,
        system_prompt: str,
        user_prompt: str | None,
        rebuild_client: bool = False,
    ):
        self.config = config
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.client.update_runtime(config, rebuild_client=rebuild_client)

    def _final_user_prompt(self, prompt_override: str | None = None) -> str:
        if prompt_override and prompt_override.strip():
            base = prompt_override.strip()
        elif self.config.mode == "unconditional":
            base = ""
        else:
            base = (self.user_prompt or "").strip()

        if self.config.max_words and self.config.max_words > 0:
            suffix = f"Respond in at most {int(self.config.max_words)} words."
            base = f"{base}\n{suffix}".strip()
        return base

    async def caption(
        self,
        image_bytes: bytes,
        *,
        prompt_override: str | None = None,
        call_overrides: dict[str, Any] | None = None,
    ) -> CaptionInferenceResult:
        prompt = self._final_user_prompt(prompt_override)
        kwargs = dict(self.config.call_kwargs or {})
        if call_overrides:
            kwargs.update(call_overrides)
        kwargs = normalize_call_kwargs(self.config.provider, kwargs)
        text = await self.client.infer(
            self.system_prompt,
            prompt,
            image_bytes,
            call_overrides=kwargs,
        )
        return CaptionInferenceResult(
            text=text.strip(),
            provider=self.config.provider,
            model_id=self.config.model_id,
        )

    async def caption_image(
        self,
        image: Image.Image,
        *,
        prompt_override: str | None = None,
        call_overrides: dict[str, Any] | None = None,
    ) -> CaptionInferenceResult:
        with io.BytesIO() as buf:
            image.save(buf, format="JPEG")
            image_bytes = buf.getvalue()
        return await self.caption(
            image_bytes, prompt_override=prompt_override, call_overrides=call_overrides
        )
