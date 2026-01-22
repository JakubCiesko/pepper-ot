import asyncio
import base64
import json
import logging
from pathlib import Path

from openai import AsyncOpenAI

from research.src.models.common import OntologyConfig
from research.src.models.data_generation import LLMLabelerConfig

logger = logging.getLogger(__name__)


class SceneGraphGenerator:
    def __init__(
        self, config: LLMLabelerConfig, ontology_config: OntologyConfig | None = None
    ):
        self.config = config
        self.client = AsyncOpenAI()
        self.ontology_config = ontology_config
        self.predicates = (
            None if self.ontology_config is None else self.ontology_config.predicates
        )

    @staticmethod
    def _encode_image(image_path: Path) -> str:
        with image_path.open("rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    async def generate(self, image_path: Path, max_retries: int = 3):
        loop = asyncio.get_running_loop()
        try:
            encoded_image = await loop.run_in_executor(
                None, self._encode_image, image_path
            )
        except Exception as e:
            logger.error(f"Failed to read/encode {image_path}: {e}")
            return []
        system_prompt = self.config.system_prompt
        user_prompt = (
            "List of allowed predicates: " + ", ".join(self.predicates)
            if self.predicates
            else "Put emphasis on spatial, functional, semantic, and action based relations."
        )
        for attempt in range(max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model_id,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{encoded_image}"
                                    },
                                },
                            ],
                        },
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    response_format={"type": "json_object"},
                )
                content_str = response.choices[0].message.content
                if not content_str:
                    return []
                data = json.loads(content_str)
                # fallback if it returns dict and not list
                if isinstance(data, dict):
                    for key in [
                        "relationships",
                        "scene_graph",
                        "triplets",
                        "data",
                        "relations",
                    ]:
                        if key in data and isinstance(data[key], list):
                            return data[key]
                    # If no known wrapper key is found, assume the dict IS the single relationship (unlikely) or wrap it
                    return [data]
                return data if isinstance(data, list) else []

            except Exception as e:
                wait_time = 2**attempt
                logger.error(
                    f"Error for {image_path.name}. Retrying in {wait_time}s... Error: {e}"
                )
                await asyncio.sleep(wait_time)
        return []

    async def batch_generate(self, image_paths: list[Path], batch_size: int = 100):
        semaphore = asyncio.Semaphore(batch_size)

        async def limited_generate(path: Path):
            async with semaphore:
                result = await self.generate(path)
                return path, result

        tasks = [limited_generate(p) for p in image_paths]
        logger.info(
            f"Starting batch generation for {len(tasks)} images with concurrency {batch_size}..."
        )
        results = await asyncio.gather(*tasks)
        return results
