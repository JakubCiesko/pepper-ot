from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from .bootstrap import ensure_server_app_importable


class ServerVLMAdapter:
    def __init__(
        self,
        provider: str,
        model_id: str,
        structured_mode: str = "provider_native",
        device: str | None = None,
        base_url: str | None = None
    ):
        ensure_server_app_importable()
        from app.providers.vlm.factory import build_vlm_client
        from app.schemas.config import LLMConfig
        from app.schemas.config import StructuredOutputConfig

        cfg = LLMConfig(
            provider=provider,
            model_id=model_id,
            device=device,
            structured_output=StructuredOutputConfig(mode=structured_mode),
            base_url=base_url
        )
        self._client = build_vlm_client(cfg)

    @staticmethod
    def _image_to_bytes(image: Image.Image) -> bytes:
        with BytesIO() as buf:
            image.convert("RGB").save(buf, format="JPEG", quality=95)
            return buf.getvalue()

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        output_schema: Any,
    ) -> tuple[str, Any | None]:
        raw_text, parsed = await self._client.infer(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image=image_bytes,
            output_schema=output_schema,
        )
        return raw_text, parsed

    async def generate_structured_from_image(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image: Image.Image,
        output_schema: Any,
    ) -> tuple[str, Any | None]:
        image_bytes = self._image_to_bytes(image)
        return await self.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_bytes=image_bytes,
            output_schema=output_schema,
        )

    async def generate_structured_from_path(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_path: str | Path,
        output_schema: Any,
    ) -> tuple[str, Any | None]:
        path = Path(image_path)
        with Image.open(path) as image:
            return await self.generate_structured_from_image(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image=image,
                output_schema=output_schema,
            )
