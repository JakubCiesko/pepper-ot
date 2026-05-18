from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from .bootstrap import ensure_server_app_importable


class ServerVLMAdapter:
    """In-process adapter around the server VLM client factory.

    Experiment workflows use this adapter for draft scene graph generation and
    context-rot runs so the same provider implementations are exercised as in
    the server pipeline.
    """

    def __init__(
        self,
        provider: str,
        model_id: str,
        structured_mode: str = "provider_native",
        device: str | None = None,
        base_url: str | None = None,
    ):
        """Create a VLM adapter from research model config fields.

        Args:
            provider: Server VLM provider name.
            model_id: Provider-specific model identifier.
            structured_mode: Structured output mode configured for the client.
            device: Optional local device hint for local backends.
            base_url: Optional OpenAI-compatible endpoint base URL.
        """
        ensure_server_app_importable()
        from app.providers.vlm.factory import build_vlm_client
        from app.schemas.config import LLMConfig
        from app.schemas.config import StructuredOutputConfig

        cfg = LLMConfig(
            provider=provider,
            model_id=model_id,
            device=device,
            structured_output=StructuredOutputConfig(mode=structured_mode),
            base_url=base_url,
        )
        self._client = build_vlm_client(cfg)

    @staticmethod
    def _image_to_bytes(image: Image.Image) -> bytes:
        """Serialize a PIL image into JPEG bytes for VLM inference."""
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
        """Generate structured VLM output from pre-encoded image bytes.

        Args:
            system_prompt: System prompt sent to the VLM.
            user_prompt: User prompt sent to the VLM.
            image_bytes: JPEG or other server-supported image bytes.
            output_schema: Pydantic schema expected from the model response.

        Returns:
            Tuple of raw model text and parsed schema instance, or None when
            parsing failed.
        """
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
        """Generate structured VLM output from a PIL image.

        Args:
            system_prompt: System prompt sent to the VLM.
            user_prompt: User prompt sent to the VLM.
            image: PIL image to encode as JPEG.
            output_schema: Pydantic schema expected from the model response.

        Returns:
            Tuple of raw model text and parsed schema instance, or None when
            parsing failed.
        """
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
        """Generate structured VLM output from an image file path.

        Args:
            system_prompt: System prompt sent to the VLM.
            user_prompt: User prompt sent to the VLM.
            image_path: Path to an image file.
            output_schema: Pydantic schema expected from the model response.

        Returns:
            Tuple of raw model text and parsed schema instance, or None when
            parsing failed.
        """
        path = Path(image_path)
        with Image.open(path) as image:
            return await self.generate_structured_from_image(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image=image,
                output_schema=output_schema,
            )
