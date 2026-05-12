import asyncio
from pathlib import Path
from typing import Any

from PIL import Image

from .bootstrap import ensure_server_app_importable
from .utils import resize_pil


class ServerCaptionAdapter:
    def __init__(self, model_provider: str, model_id: str, system_prompt: str, base_url: str | None = None):
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
            base_url=base_url
        )
        self._service = CaptionInferenceService(
            cfg,
            system_prompt=system_prompt,
            user_prompt=None,
        )

    async def caption_image(
        self,
        image: Image.Image,
        prompt_override: str | None = None,
        max_image_size: int | None = None,
    ) -> dict[str, Any]:
        if max_image_size:
            image = resize_pil(image, max_image_size)
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
        max_image_size: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        semaphore = asyncio.Semaphore(max_concurrent)
        output: dict[str, dict[str, Any]] = {}

        async def run_one(path: Path):
            async with semaphore:
                with Image.open(path) as img:
                    image = img.convert("RGB")
                prompt = prompt_builder(path)
                caption_payload = await self.caption_image(
                    image, prompt_override=prompt, max_image_size=max_image_size
                )
                output[str(path.resolve())] = caption_payload

        await asyncio.gather(*(run_one(path) for path in image_paths))
        return output
