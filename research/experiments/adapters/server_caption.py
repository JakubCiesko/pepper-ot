import asyncio
from pathlib import Path
from typing import Any

from PIL import Image

from .bootstrap import ensure_server_app_importable


class ServerCaptionAdapter:
    def __init__(self, model_provider: str, model_id: str, system_prompt: str):
        ensure_server_app_importable()
        from app.inference.caption.service import CaptionInferenceService
        from app.schemas.config import CaptionConfig
        from app.schemas.config import PromptSource

        cfg = CaptionConfig(
            provider=model_provider,
            model_id=model_id,
            mode="prompted",
            system_prompt=PromptSource(text=system_prompt),
            user_prompt=None,
        )
        self._service = CaptionInferenceService(
            cfg,
            system_prompt=system_prompt,
            user_prompt=None,
        )

    async def caption_image(
        self, image: Image.Image, prompt_override: str | None = None
    ) -> dict[str, Any]:
        result = await self._service.caption_image(
            image, prompt_override=prompt_override
        )
        if hasattr(result, "model_dump"):
            return result.model_dump()
        return result.__dict__

    async def caption_images(
        self,
        image_paths: list[Path],
        prompt_builder,
        max_concurrent: int = 4,
    ) -> dict[str, dict[str, Any]]:
        semaphore = asyncio.Semaphore(max_concurrent)
        output: dict[str, dict[str, Any]] = {}

        async def run_one(path: Path):
            async with semaphore:
                with Image.open(path) as img:
                    image = img.convert("RGB")
                prompt = prompt_builder(path)
                caption_payload = await self.caption_image(
                    image, prompt_override=prompt
                )
                output[str(path)] = caption_payload

        await asyncio.gather(*(run_one(path) for path in image_paths))
        return output
