import asyncio
import base64
import logging
import time
from typing import TYPE_CHECKING
from uuid import uuid4

from app.providers.caption_client import CaptionClient
from app.providers.translation import enforce_output_language
from app.schemas.config import CaptionConfig
from app.schemas.robot import RobotMetadata

if TYPE_CHECKING:
    from app.core.runtime.state import AppState

logger = logging.getLogger(__name__)


class CaptionService:
    def __init__(
        self,
        state: "AppState",
        config: CaptionConfig,
        *,
        system_prompt: str,
        user_prompt: str | None,
    ):
        self.state = state
        self.config = config
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.client: CaptionClient | None = None

    def _ensure_client(self):
        if self.client is None:
            self.client = CaptionClient(self.config)

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
        if self.client is None:
            if rebuild_client:
                self.client = CaptionClient(config)
            return
        self.client.update_runtime(config, rebuild_client=rebuild_client)

    def _final_user_prompt(self, prompt_override: str | None) -> str:
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
        language: str | None = None,
    ) -> str:
        prompt = self._final_user_prompt(prompt_override)
        output_language = (
            language
            if language is not None
            else (
                self.state.config.system.get("output_language")
                if self.state.config is not None
                and isinstance(self.state.config.system, dict)
                else None
            )
        )
        if (
            self.state.config is not None
            and self.state.config.worker.enabled
            and self.state.worker_manager is not None
        ):
            payload = await self.state.worker_manager.request(
                "POST",
                "/internal/caption",
                json={
                    "image_b64": base64.b64encode(image_bytes).decode("utf-8"),
                    "prompt": prompt,
                },
            )
            text = str(payload.get("caption", "")).strip()
            return await enforce_output_language(text, output_language)
        self._ensure_client()
        assert self.client is not None
        text = await self.client.infer(self.system_prompt, prompt, image_bytes)
        return await enforce_output_language(text, output_language)

    async def caption_with_optional_detect(
        self,
        image_bytes: bytes,
        *,
        metadata: RobotMetadata,
        run_detect: bool,
        publish: bool,
        prompt_override: str | None = None,
        language: str | None = None,
    ) -> dict:
        caption_text = await self.caption(
            image_bytes, prompt_override=prompt_override, language=language
        )
        detect_request_id: str | None = None
        if run_detect:
            detect_request_id = str(uuid4())
            asyncio.create_task(
                self._run_detect_background(
                    request_id=detect_request_id,
                    image_bytes=image_bytes,
                    metadata=metadata,
                    publish=publish,
                )
            )
        return {
            "caption": caption_text,
            "provider": self.config.provider,
            "model_id": self.config.model_id,
            "detect_started": run_detect,
            "detect_request_id": detect_request_id,
            "timestamp": time.time(),
        }

    async def _run_detect_background(
        self,
        *,
        request_id: str,
        image_bytes: bytes,
        metadata: RobotMetadata,
        publish: bool,
    ):
        try:
            from app.orchestration.detect_service import DetectService

            await DetectService(self.state).process(
                image_bytes, metadata, publish=publish
            )
            logger.info(
                "Caption detect background job finished request_id=%s", request_id
            )
        except Exception:
            logger.exception(
                "Caption detect background job failed request_id=%s", request_id
            )
