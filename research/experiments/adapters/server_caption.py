import asyncio
from pathlib import Path
from typing import Any

from PIL import Image

from .bootstrap import ensure_server_app_importable
from .utils import resize_pil


class ServerCaptionAdapter:
    """In-process adapter around the server caption inference service.

    The description phase uses this adapter to apply research prompts while
    sharing the same caption provider configuration path as the server.
    """

    def __init__(
        self,
        model_provider: str,
        model_id: str,
        system_prompt: str,
        base_url: str | None = None,
    ):
        """Create a caption adapter from research model and prompt settings.

        Args:
            model_provider: Caption provider name understood by the server.
            model_id: Provider-specific model identifier.
            system_prompt: System prompt used by the caption service.
            base_url: Optional OpenAI-compatible endpoint base URL.
        """
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
            base_url=base_url,
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
        """Caption one image with an optional per-image prompt override.

        Args:
            image: PIL image to caption.
            prompt_override: Optional user prompt built by the description
                workflow.
            max_image_size: Optional longest-side resize limit before
                inference.

        Returns:
            Caption payload as a dictionary, typically including generated text
            and provider metadata.
        """
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
        """Caption multiple image paths concurrently.

        Args:
            image_paths: Image paths to load and caption.
            prompt_builder: Callable that receives each Path and returns the
                user prompt override.
            max_concurrent: Maximum concurrent caption requests.
            max_image_size: Optional longest-side resize limit before
                inference.

        Returns:
            Mapping from resolved image path to caption payload.
        """
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
